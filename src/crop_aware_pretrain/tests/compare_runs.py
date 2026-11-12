"""Compare train.csv rows across verification runs: A world1, B resumed world1, C world2, D 2->1, E 1->2."""
import csv, sys
V = sys.argv[1]
def rows(d):
    out = {}
    for r in csv.DictReader(open(f"{V}/{d}/logs/train.csv")):
        out[int(r["step"])] = r
    return out
A = rows("A")
ok = True
for name, ref in (("B", "A"), ("C", "A"), ("D", "A"), ("E", "A")):
    R = rows(name)
    cmp_steps = sorted(s for s in A if s in R and s != 1)
    worst = {k: 0.0 for k in ("loss", "lr_top", "l_pool", "l_preserve", "roi_pool_cos")}
    for s in cmp_steps:
        for k in worst:
            worst[k] = max(worst[k], abs(float(A[s][k]) - float(R[s][k])))
    print(name, "vs", ref, "steps compared", cmp_steps[0], "..", cmp_steps[-1], "n", len(cmp_steps), {k: f"{v:.2e}" for k, v in worst.items()}, "world/accum at end", R[cmp_steps[-1]]["world"], R[cmp_steps[-1]]["accum"])
    ok &= worst["lr_top"] == 0 and worst["loss"] < 2e-3
print("OK" if ok else "MISMATCH")
