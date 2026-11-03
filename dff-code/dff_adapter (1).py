"""
dff_adapter.py -- DeepFake Fine-Grained Adapter (DFF-Adapter) on a frozen DINOv2.

Paper: "Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection"
(arXiv:2511.12107). Equation numbers below refer to that paper.

HOW IT FITS TOGETHER (read this first)
--------------------------------------
A DINOv2 transformer block has three linear layers we adapt:
    attention.query   (d -> d)     attention.value  (d -> d)     attention.output.dense (d -> d)
Each is replaced by a DFFLinear = frozen original linear + a DFFAdapter:
    y = W x  +  scale * adapter(x)           scale = alpha / rank = 32 / 16 = 2   (the paper's beta)

Inside one DFFAdapter (H = 4 "heads", N = 6 experts per head):
    1. split the d input channels into H chunks of width d_h = d / H
    2. chunk 0 -> SHARED head   : mixes ALL N experts with softmax weights   (Eq. 5)
       chunks 1..H-1 -> TASK heads: each mixes its TOP-3 of N experts with
                                    renormalised softmax weights             (Eq. 1-4)
       the task heads use routing logits Z_task[task] -> the same experts,
       different routing for task 0 (authenticity) vs task 1 (forgery type).
    3. an expert is a tiny LoRA pair: A (d_h -> r_h), B (r_h -> d_h), r_h = rank / H = 4
    4. concatenate the H head outputs back to width d  (Eq. 6 / 8: "concat(shared, task heads)")
So every input chunk is used exactly once, and the shared head's output is common to both tasks
(that is how forgery-type knowledge flows into the authenticity branch: they share chunk 0's
experts, and every frozen layer sees updates trained by both losses).

EFFICIENCY NOTE: the paper's routing logits Z are learned tensors, not functions of the input,
so the mixing weights are the same for every token. We therefore compute all experts of a head
with one batched einsum and multiply by a (H, N) weight matrix that is zero for unselected
experts. This is mathematically identical to looping over the top-3 experts (the unit test
test_matches_naive_equations checks this against a literal implementation of Eq. 1-5).
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DFFConfig:
    rank: int = 16               # r      (paper: 16)
    alpha: float = 32.0          # alpha  (paper: 32)  -> scale beta = alpha / rank
    n_experts: int = 6           # N      (paper: 6)
    n_heads: int = 4             # H = 1 shared + (H-1) task heads (paper: 4)
    top_k: int = 3               # experts activated per task head (paper: 3)
    n_tasks: int = 2             # 0 = authenticity (BCE), 1 = forgery type (CE)
    router_init_std: float = 0.5  # OUR choice: std of random init of routing logits


class TaskContext:
    """Mutable holder of the current task id, shared by every adapter in a model."""
    def __init__(self):
        self.task = 0


class DFFAdapter(nn.Module):
    def __init__(self, d_in, d_out, cfg: DFFConfig, ctx: TaskContext):
        super().__init__()
        assert d_in == d_out, "per-head chunking needs d_in == d_out (true for q, v, dense)"
        H, N = cfg.n_heads, cfg.n_experts
        assert d_in % H == 0, f"d={d_in} must be divisible by n_heads={H}"
        assert cfg.rank % H == 0, f"rank={cfg.rank} must be divisible by n_heads={H}"
        assert 1 <= cfg.top_k <= N
        self.cfg, self.ctx = cfg, ctx
        self.d_h, self.r_h = d_in // H, cfg.rank // H
        self.scale = cfg.alpha / cfg.rank

        # expert banks: one bank of N experts per head.  A: (H,N,r_h,d_h)  B: (H,N,d_h,r_h)
        self.A = nn.Parameter(torch.empty(H, N, self.r_h, self.d_h))
        self.B = nn.Parameter(torch.zeros(H, N, self.d_h, self.r_h))  # zeros => adapter starts as identity
        bound = 1.0 / math.sqrt(self.d_h)                              # = kaiming_uniform(a=sqrt(5)) bound
        nn.init.uniform_(self.A, -bound, bound)

        # routing logits (learned tensors, Eq. 4 and 5)
        self.z_task = nn.Parameter(torch.randn(cfg.n_tasks, H - 1, N) * cfg.router_init_std)
        self.z_shared = nn.Parameter(torch.randn(N) * cfg.router_init_std)

    def gate_weights(self, task=None):
        """(H, N) mixing weights. Row 0 = shared head (all N experts, softmax, Eq. 5).
        Rows 1..H-1 = task heads (softmax -> top-k -> renormalise, Eq. 2-4); zeros elsewhere."""
        t = self.ctx.task if task is None else task
        p = F.softmax(self.z_task[t].float(), dim=-1)                  # Eq. 4   (H-1, N)
        top_v, top_i = p.topk(self.cfg.top_k, dim=-1)                  # Eq. 3
        w_task = torch.zeros_like(p).scatter(-1, top_i, top_v / top_v.sum(-1, keepdim=True))  # Eq. 2
        w_shared = F.softmax(self.z_shared.float(), dim=-1)[None]      # (1, N)
        return torch.cat([w_shared, w_task], dim=0)

    def forward(self, x):                                              # x: (batch, tokens, d)
        b, l, d = x.shape
        H, N = self.cfg.n_heads, self.cfg.n_experts
        xc = x.reshape(b, l, H, self.d_h)                              # split channels into H chunks
        z = torch.einsum("blhd,hnrd->blhnr", xc, self.A)               # A_j x     for all experts
        w = self.gate_weights().to(z.dtype)                            # (H, N)
        z = z * w[None, None, :, :, None]                              # zero out unselected experts
        out = torch.einsum("blhnr,hndr->blhd", z, self.B)              # sum_j g_j B_j (A_j x)
        return out.reshape(b, l, d) * self.scale                       # concat heads, times beta


class PlainLoRAAdapter(nn.Module):
    """BASELINE (paper Table 4 'LoRA'): standard LoRA, y = Wx + (alpha/r) * B(A x).
    One expert, rank r, no routing, no heads. Same cfg.rank / cfg.alpha as the DFF-Adapter."""
    def __init__(self, d_in, d_out, cfg: DFFConfig, ctx: TaskContext):
        super().__init__()
        self.cfg, self.ctx = cfg, ctx
        self.scale = cfg.alpha / cfg.rank
        self.A = nn.Parameter(torch.empty(cfg.rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, cfg.rank))        # zeros => starts as a no-op
        bound = 1.0 / math.sqrt(d_in)
        nn.init.uniform_(self.A, -bound, bound)

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B) * self.scale


class DFFLinear(nn.Module):
    """Frozen linear layer + adapter:  y = W x + adapter(x).  kind: 'dff' (ours) or 'lora' (baseline)."""
    def __init__(self, base: nn.Linear, cfg: DFFConfig, ctx: TaskContext, kind="dff"):
        super().__init__()
        self.base = base
        adapter_cls = DFFAdapter if kind == "dff" else PlainLoRAAdapter
        self.adapter = adapter_cls(base.in_features, base.out_features, cfg, ctx)

    def forward(self, x):
        return self.base(x) + self.adapter(x)


def inject_dff(backbone, cfg: DFFConfig, kind="dff"):
    """Freeze `backbone` and wrap query / value / attention-output-dense of EVERY block.
    This is THE function that "adds the adapters to the transformer".
    kind: 'dff' = our adapter, 'lora' = plain-LoRA baseline, 'none' = linear probe (nothing injected).
    Module paths verified for transformers 4.57.6 and 5.16.1 (Dinov2Layer)."""
    assert kind in ("dff", "lora", "none"), kind
    for p in backbone.parameters():
        p.requires_grad_(False)
    ctx, adapters = TaskContext(), []
    if kind == "none":
        return ctx, adapters
    for i, layer in enumerate(backbone.encoder.layer):
        targets = [(layer.attention.attention, "query"),
                   (layer.attention.attention, "value"),
                   (layer.attention.output, "dense")]
        for owner, name in targets:
            base = getattr(owner, name, None)
            if not isinstance(base, nn.Linear):
                raise RuntimeError(f"block {i}: expected nn.Linear at .{name}, got {type(base)}. "
                                   "Module layout differs in this transformers version -- print the layer.")
            wrapped = DFFLinear(base, cfg, ctx, kind)  # created AFTER freezing => its params are trainable
            setattr(owner, name, wrapped)
            adapters.append(wrapped.adapter)
    return ctx, adapters


class DFFDetector(nn.Module):
    """Frozen DINOv2 + DFF adapters + two linear heads on the CLS token.
       forward(images, task=0) -> authenticity logit (batch,)      [BCE-with-logits]
       forward(images, task=1) -> forgery-type logits (batch, C)   [cross-entropy]
    Training does BOTH passes on the same batch (paper, 'Training Details');
    inference uses task=0 only."""
    def __init__(self, backbone, cfg: DFFConfig = None, n_forgery_classes: int = 5, kind: str = "dff"):
        super().__init__()
        self.cfg = cfg or DFFConfig()
        self.backbone = backbone
        self.ctx, self.adapters = inject_dff(backbone, self.cfg, kind)
        d = backbone.config.hidden_size
        self.head_auth = nn.Linear(d, 1)
        self.head_type = nn.Linear(d, n_forgery_classes)

    def forward(self, pixel_values, task=0):
        self.ctx.task = task
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]   # CLS token
        return self.head_auth(cls).squeeze(-1) if task == 0 else self.head_type(cls)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ------------------------------------------------------------------ diagnostics
def expected_adapter_params(d, n_layers, cfg: DFFConfig):
    """Closed-form count of adapter parameters (3 wrapped linears per block)."""
    H, N = cfg.n_heads, cfg.n_experts
    d_h, r_h = d // H, cfg.rank // H
    per_adapter = H * N * (2 * r_h * d_h) + cfg.n_tasks * (H - 1) * N + N
    return per_adapter * 3 * n_layers


@torch.no_grad()
def routing_report(adapters, cfg: DFFConfig):
    """How many (adapter, head, expert) slots are never selected by ANY task?
    An expert that is never selected gets zero gradient and never trains ('dead')."""
    H, N = cfg.n_heads, cfg.n_experts
    used = torch.zeros(len(adapters), H, N, dtype=torch.bool)
    for i, a in enumerate(adapters):
        for t in range(cfg.n_tasks):
            used[i] |= (a.gate_weights(t).cpu() > 0)
    dead = int((~used).sum())
    return {"slots": used.numel(), "dead": dead, "dead_fraction": dead / used.numel()}
