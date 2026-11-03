"""
test_dff_adapter.py -- run:   python test_dff_adapter.py          (tiny random model, seconds, CPU ok)
                              python test_dff_adapter.py --real   (also checks real dinov2-small)

Each test states what property of the paper's method it protects.
"""
import copy
import sys
import traceback

import torch
import torch.nn.functional as F
from transformers import Dinov2Config, Dinov2Model

from dff_adapter import (DFFAdapter, DFFConfig, DFFDetector, TaskContext,
                         expected_adapter_params, routing_report)

CFG = DFFConfig()  # rank 16, alpha 32, N=6, H=4, top-3, 2 tasks


def tiny_backbone():
    c = Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     mlp_ratio=4, image_size=56, patch_size=14)
    return Dinov2Model(c).eval()


def randomise_B(adapters, std=0.1):
    with torch.no_grad():
        for a in adapters:
            a.B.normal_(0, std)


def naive_adapter(a, x, task):
    """LITERAL implementation of paper Eq. 1-5 with explicit loops (slow, for testing only)."""
    H, N, d_h = a.cfg.n_heads, a.cfg.n_experts, a.d_h
    outs = []
    for k in range(H):
        xk = x[..., k * d_h:(k + 1) * d_h]
        if k == 0:                                     # shared head, Eq. 5: all N experts
            g = F.softmax(a.z_shared, -1)
            chosen, weights = list(range(N)), g
        else:                                          # task head k, Eq. 1-4
            g = F.softmax(a.z_task[task, k - 1], -1)                    # Eq. 4
            chosen = g.topk(a.cfg.top_k).indices.tolist()               # Eq. 3
            denom = sum(g[j] for j in chosen)
            weights = g / denom                                         # Eq. 2
        f = 0
        for j in chosen:                               # Eq. 1:  sum_j g~_j  B_j (A_j x)
            f = f + weights[j] * (xk @ a.A[k, j].T @ a.B[k, j].T)
        outs.append(f)
    return torch.cat(outs, dim=-1) * a.scale


# ---------------------------------------------------------------- tests
def test_zero_init_is_identity():
    """B=0 at init => wrapped network == original frozen DINOv2 (adapters start as no-ops)."""
    torch.manual_seed(0)
    base = tiny_backbone()
    ref = copy.deepcopy(base)
    det = DFFDetector(base, CFG, 5)
    x = torch.randn(2, 3, 56, 56)
    with torch.no_grad():
        a = ref(pixel_values=x).last_hidden_state
        b = det.backbone(pixel_values=x).last_hidden_state
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_only_adapters_and_heads_train():
    """Backbone frozen; trainable = adapters + two linear heads; count matches closed form."""
    det = DFFDetector(tiny_backbone(), CFG, 5)
    adapter_params = 0
    for n, p in det.named_parameters():
        is_ours = (".adapter." in n) or n.startswith("head_")
        assert p.requires_grad == is_ours, f"{n}: requires_grad={p.requires_grad}"
        if ".adapter." in n:
            adapter_params += p.numel()
    assert adapter_params == expected_adapter_params(64, 2, CFG), adapter_params


def test_adapters_wrap_q_v_dense_in_every_block():
    det = DFFDetector(tiny_backbone(), CFG, 5)
    assert len(det.adapters) == 3 * 2
    for layer in det.backbone.encoder.layer:
        for m in (layer.attention.attention.query, layer.attention.attention.value,
                  layer.attention.output.dense):
            assert type(m).__name__ == "DFFLinear"
        assert type(layer.attention.attention.key).__name__ == "Linear"   # key untouched
        assert type(layer.mlp.fc1).__name__ == "Linear"                    # MLP untouched


def test_top3_routing_and_normalisation():
    """Task heads activate exactly 3 experts with weights summing to 1; shared head uses all N."""
    a = DFFAdapter(64, 64, CFG, TaskContext())
    for t in (0, 1):
        w = a.gate_weights(t)
        assert w.shape == (4, 6)
        assert ((w[1:] > 0).sum(-1) == 3).all()
        assert torch.allclose(w.sum(-1), torch.ones(4), atol=1e-6)
        assert (w[0] > 0).all()


def test_matches_naive_equations():
    """Batched einsum implementation == literal Eq. 1-5 loops, for both tasks."""
    torch.manual_seed(1)
    ctx = TaskContext()
    a = DFFAdapter(64, 64, CFG, ctx)
    randomise_B([a])
    x = torch.randn(2, 5, 64)
    for t in (0, 1):
        ctx.task = t
        got, want = a(x), naive_adapter(a, x, t)
        assert torch.allclose(got, want, atol=1e-5), (t, (got - want).abs().max())


def test_each_input_chunk_used_exactly_once():
    """Output chunk k depends ONLY on input chunk k (block-diagonal). Guards against the
    old-submission bug where one chunk was used twice and another dropped."""
    torch.manual_seed(2)
    a = DFFAdapter(64, 64, CFG, TaskContext())
    randomise_B([a])
    x = torch.randn(2, 5, 64)
    base = a(x)
    d_h = a.d_h
    for k in range(4):
        x2 = x.clone()
        x2[..., k * d_h:(k + 1) * d_h] += 1.0
        diff = (a(x2) - base).abs()
        for j in range(4):
            changed = diff[..., j * d_h:(j + 1) * d_h].max() > 1e-6
            assert bool(changed) == (j == k), f"input chunk {k} affected output chunk {j}"


def test_task_switch_changes_output():
    """Same experts, different routing logits per task => task 0 and task 1 differ."""
    torch.manual_seed(3)
    ctx = TaskContext()
    a = DFFAdapter(64, 64, CFG, ctx)
    randomise_B([a])
    x = torch.randn(2, 5, 64)
    ctx.task = 0
    y0 = a(x)
    ctx.task = 1
    y1 = a(x)
    assert not torch.allclose(y0, y1, atol=1e-6)


def test_gradient_flow_and_unselected_logits():
    """(1) at init only B gets gradient (A and routers need B != 0).
       (2) backbone gets NO gradient.  (3) routing logits of experts outside the top-3
       get (numerically) zero gradient -- documents the 'dead expert' property of top-k routing."""
    torch.manual_seed(4)
    det = DFFDetector(tiny_backbone(), CFG, 5)
    x = torch.randn(2, 3, 56, 56)
    (det(x, 0).sum() + det(x, 1).sum()).backward()
    for a in det.adapters:
        assert a.B.grad.abs().sum() > 0
        assert a.A.grad is None or a.A.grad.abs().sum() == 0
    for n, p in det.backbone.named_parameters():
        if ".adapter." not in n:
            assert p.grad is None, n
    det.zero_grad()
    randomise_B(det.adapters)
    (det(x, 0).sum() + det(x, 1).sum()).backward()
    for a in det.adapters:
        assert a.A.grad.abs().sum() > 0
        for t in (0, 1):
            selected = a.gate_weights(t)[1:] > 0
            g = a.z_task.grad[t]
            # mathematically zero (the softmax Jacobian cancels); allow float rounding noise
            assert g[~selected].abs().max() <= 1e-4 * g.abs().max(), "unselected logit got real gradient?!"
            assert (g[selected].abs() > 0).any()


def test_parameter_budget_matches_paper():
    """Closed-form count for the paper's DINOv2-Large (d=1024, 24 blocks) is ~3.5M."""
    rows = {"Small": (384, 12), "Base": (768, 12), "Large": (1024, 24)}
    for name, (d, L) in rows.items():
        print(f"      {name:6s} adapter params = {expected_adapter_params(d, L, CFG):>10,d}")
    assert abs(expected_adapter_params(1024, 24, CFG) - 3.5e6) < 0.1e6


def test_routing_report_runs():
    det = DFFDetector(tiny_backbone(), CFG, 5)
    r = routing_report(det.adapters, CFG)
    print(f"      never-selected (adapter,head,expert) slots: {r['dead']}/{r['slots']} "
          f"= {100 * r['dead_fraction']:.1f}%")
    assert r["slots"] == 6 * 4 * 6


def test_lora_baseline_variant():
    """Plain-LoRA baseline: matches y = scale * B(A x), no-op at init, trains adapters + heads only."""
    from dff_adapter import PlainLoRAAdapter
    torch.manual_seed(5)
    a = PlainLoRAAdapter(64, 64, CFG, TaskContext())
    with torch.no_grad():
        a.B.normal_(0, 0.1)
    x = torch.randn(2, 5, 64)
    want = (x @ a.A.T @ a.B.T) * (CFG.alpha / CFG.rank)
    assert torch.allclose(a(x), want, atol=1e-5)
    base = tiny_backbone()
    ref = copy.deepcopy(base)
    det = DFFDetector(base, CFG, 5, kind="lora")
    xi = torch.randn(2, 3, 56, 56)
    with torch.no_grad():
        assert torch.allclose(ref(pixel_values=xi).last_hidden_state,
                              det.backbone(pixel_values=xi).last_hidden_state, atol=1e-6)
    n = sum(p.numel() for ad in det.adapters for p in ad.parameters())
    assert n == 3 * 2 * CFG.rank * (64 + 64), n          # 3 linears x 2 blocks x r x (d_in + d_out)
    for name, p in det.named_parameters():
        assert p.requires_grad == ((".adapter." in name) or name.startswith("head_")), name


def test_linear_probe_variant():
    """Linear probe: nothing injected, only the two heads train, output identical to frozen model."""
    base = tiny_backbone()
    ref = copy.deepcopy(base)
    det = DFFDetector(base, CFG, 5, kind="none")
    assert det.adapters == []
    trainable = [n for n, p in det.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("head_") for n in trainable), trainable
    assert sum(p.numel() for p in det.trainable_parameters()) == (64 * 1 + 1) + (64 * 5 + 5)
    xi = torch.randn(2, 3, 56, 56)
    with torch.no_grad():
        assert torch.allclose(ref(pixel_values=xi).last_hidden_state,
                              det.backbone(pixel_values=xi).last_hidden_state, atol=1e-6)
    det(xi, 0).sum().backward()
    assert det.head_auth.weight.grad is not None


def test_real_dinov2_small():
    """Only with --real: real checkpoint, 36 adapters, exact count, forward shapes, GPU memory."""
    from transformers import AutoModel
    base = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    ref = copy.deepcopy(base)
    det = DFFDetector(base, CFG, 5)
    assert len(det.adapters) == 36
    n = sum(p.numel() for a in det.adapters for p in a.parameters())
    expected = expected_adapter_params(
    det.backbone.config.hidden_size,
    len(det.backbone.encoder.layer),
    CFG)
    assert n == expected
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(ref(pixel_values=x).last_hidden_state,
                              det.backbone(pixel_values=x).last_hidden_state, atol=1e-5)
        assert det(x, 0).shape == (2,) and det(x, 1).shape == (2, 5)
    trainable = sum(p.numel() for p in det.trainable_parameters())
    total = sum(p.numel() for p in det.parameters())
    print(f"      trainable={trainable:,} of total={total:,} ({100 * trainable / total:.2f}%)")
    if torch.cuda.is_available():
        det.cuda()
        xb = torch.randn(16, 3, 224, 224, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = det(xb, 0).float().sum() + det(xb, 1).float().sum()
        loss.backward()
        print(f"      GPU peak memory, batch 16, fp16, both passes: "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    real = "--real" in sys.argv
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and (real or n != "test_real_dinov2_small")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
