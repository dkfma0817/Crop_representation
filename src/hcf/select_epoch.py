"""R2 epoch selection on the official PACO VAL GT-ROI attribute AP: max (AP_obj + AP_opart) / 2, ties -> earlier epoch. Test is not read here."""
import glob
import json
import os
import sys

run = sys.argv[1]
out, curves = {}, {}
for d in sorted(glob.glob(f"{run}/eval_val/*")):
    n = os.path.basename(d)
    if n.endswith("_nofine"):
        continue
    c = {}
    for e in range(1, 11):
        f = f"{d}/ep{e}/official_results.json"
        h = json.load(open(f))["hierarchical"]
        c[e] = dict(obj=h["obj-attr-AP"], part=h["obj-attr-part-AP"])
    best = max(c, key=lambda e: (round((c[e]["obj"] + c[e]["part"]) / 2, 10), -e))
    out[n] = dict(epoch=best, val_obj=c[best]["obj"], val_part=c[best]["part"], rule="max (obj+part)/2 on official val, ties earlier")
    curves[n] = c
out["P_r2fused_nofine"] = dict(epoch=out["P_r2fused"]["epoch"], rule="same epoch as P_r2fused")
json.dump(out, open(f"{run}/selection.json", "w"), indent=1)
json.dump(curves, open(f"{run}/val_curves.json", "w"), indent=1)
print(json.dumps(out, indent=1))
