import json, glob
for ds in ["ffpp", "cdf1", "cdf2"]:
    tot = fill = 0
    frames, settings = [], set()
    for p in glob.glob(f"crops/{ds}/*/*/_status.json"):
        m = json.load(open(p))
        tot += m["n_saved"]; fill += m["n_no_detection"]
        frames.append(m["n_saved"]); settings.add((m["scale"], m["size"]))
    print(f"{ds}: videos={len(frames)} frames={tot} min/video={min(frames)} "
          f"filled={fill} ({100*fill/tot:.2f}%) settings={settings}")
