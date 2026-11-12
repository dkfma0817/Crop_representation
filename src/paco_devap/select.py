"""Official-protocol Dev AP checkpoint selection (CPU). Pre-registered rule (fixed 2026-11-07, before any use):
  score(epoch) = (Dev Object attribute AP + Dev Object-Part attribute AP) / 2   [official evaluator, dev 1,000 images]
  selected     = argmax over every evaluated epoch, ties -> the earlier epoch
Reads <preds>/<name>/ep*/official_dev.json (src.paco_devap.official_dev) and writes <preds>/<name>/selected_epoch_devap.json.
The official TEST is never read here; it is used only for the final report of the selected checkpoint.

  python -m src.paco_devap.select --name <name> [--preds output/2026-11-07/paco_devap/preds]
"""
import argparse
import json
import os

RULE = "max over epochs of (Dev Obj AP + Dev Part AP) / 2 (Official-protocol Dev AP), ties -> earlier epoch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--preds", default="output/2026-11-07/paco_devap/preds")
    a = ap.parse_args()
    d = os.path.join(a.preds, a.name)
    rec = {}
    for e in sorted(os.listdir(d)):
        f = os.path.join(d, e, "official_dev.json")
        if e.startswith("ep") and os.path.exists(f):
            o = json.load(open(f))
            rec[int(e[2:])] = dict(obj=o["obj"], part=o["part"], score=(o["obj"] + o["part"]) / 2, obj_types=o["obj_types"], part_types=o["part_types"])
    assert rec, f"no official_dev.json under {d}"
    best = max(rec, key=lambda e: (rec[e]["score"], -e))
    out = dict(rule=RULE, selected_epoch=best, selected=rec[best], per_epoch=rec, n_epochs=len(rec))
    json.dump(out, open(os.path.join(d, "selected_epoch_devap.json"), "w"), indent=1)
    print(json.dumps({"name": a.name, "selected_epoch": best, **{k: round(v, 2) for k, v in rec[best].items() if k in ("obj", "part", "score")}}))


if __name__ == "__main__":
    main()
