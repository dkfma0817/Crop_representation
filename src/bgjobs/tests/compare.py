import csv, sys
V = sys.argv[1]
def rows(d):
    return {int(r["step"]): r for r in csv.DictReader(open(f"{V}/{d}/logs/train.csv"))}
a, b = rows("B60"), rows("B30")
ss = [s for s in a if s in b and s not in (30,)]
print("B straight vs B stop+resume", "steps", ss[:3], "..", ss[-1], "max|dloss|", max(abs(float(a[s]["loss"]) - float(b[s]["loss"])) for s in ss), "lr equal", all(a[s]["lr_enc"] == b[s]["lr_enc"] for s in ss))
