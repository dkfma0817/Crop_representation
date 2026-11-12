"""PACO val evaluation of the exported EMA encoders. PRIMARY METRIC = official PACO attribute AP (GT box + GT category oracle, src.pacoattr.official_eval), Overall = (Object + Part) / 2;
the reader epoch is max (Object + Part) / 2 on official Val (src.siglip_pm.select_epoch). Project M2/M0 (`report`) is kept only as a secondary record. Test is never read here.

  python -m src.crop_aware_pretrain.evals run --step 5000   (one GPU via CUDA_VISIBLE_DEVICES; R0 + R2 are retrained from scratch: src.g1x.probe, mode 'full')
  python -m src.crop_aware_pretrain.evals report            (CPU; project M2/M0: evals/summary.csv + summary.md + control/stop_decision.json)
  python -m src.crop_aware_pretrain.evals official --step 5000 | --all   (official AP: evals/official_summary.md / .csv; one GPU for inference, CPU for the evaluator)

Protocol = probe_frozen_full (g1_cross_distill): frozen encoder, full image once, R2 full-canvas geometry, 10 epochs, PACO dev 1,000 images for the reader
epoch (max dev M0), final metrics on PACO val. Baselines: step 0 of this run (= pretrained DINOv3, also cross-checked against the existing Frozen Full probe) and
the existing C0 crop oracle probe (ivc_v3/probe_c0, same probe protocol). Object D5-16 and Part D1-4 are reported in M0 because M2 is not defined on
those subsets of val (src.readout.evaluate: too few eligible units, never imputed); M2 is given for the subset when finite.
"""
import argparse
import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

from src.crop_aware_pretrain import ckpt as CK
from src.crop_aware_pretrain import config as CF

OVERRIDES = []
PROBE = dict(name=None, backbone="dinov3", mode="full", seed=0, epochs=10, paco_dev_images=1000,
             readers=[dict(name="r0", kind="r0"), dict(name="r2", kind="r2", layers=[12])])


def step_dir(cfg, step):
    return os.path.join(CF.dirs(cfg)["evals"], f"step_{step:08d}")


OFFICIAL_DATA, EPY = "output/2026-11-07/paco_oracle_attr/data", "/home/irteam/envs/paco_eval/bin/python"
OFFICIAL_PATH = f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{CF.ROOT}"


def official_root(cfg, step):
    return os.path.join(step_dir(cfg, step), "official")


def official_infer(cfg, step, workers=8):
    """All official Val ROIs (31,919 rows) through the exported EMA encoder + each saved R2 reader epoch; full-image encoder once per image, R2 full-canvas geometry (= the probe).
    Asserts the scores reproduce the probe's saved Val scores (Pearson > 0.999), so the row alignment and the reader input are verified."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    import src.readout.common as RC
    from src.g1x.data import PacoDS
    from src.ivc.common import EXCL
    from src.readout.models import Reader
    from src.viz.c0_pca import load_encoder
    dev = torch.device("cuda")
    d, root = step_dir(cfg, step), official_root(cfg, step)
    probe = os.path.join(d, "probe", "r2")
    M = pd.read_parquet(os.path.join(OFFICIAL_DATA, "manifest_val.parquet"))
    vit = load_encoder(cfg["backbone"], dev)
    vit.load_state_dict(torch.load(os.path.join(d, "enc_ema.pt"), map_location="cpu")["enc"])
    vit.eval().requires_grad_(False)
    readers = {}
    for e in range(1, 11):
        r = Reader("r2", layers=(12,)).to(dev)
        r.load_state_dict(torch.load(os.path.join(probe, f"ep{e}.pt"), map_location=dev, weights_only=False)["reader"])
        readers[e] = r.eval().requires_grad_(False)
    ds = PacoDS(M, np.zeros((len(M), 59), np.int8), False, "full", 384, 16)
    L = {e: torch.full((len(M), 59), float("nan")) for e in readers}
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    with torch.no_grad():
        for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=workers, collate_fn=RC.collate_one, pin_memory=True):
            b = RC.to_dev(b, dev)
            toks = []
            for i in range(0, len(b["x"]), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    toks.append(vit.forward_features(b["x"][i:i + 64])[:, vit.num_prefix_tokens:].float())
            f = {12: torch.cat(toks)}
            rows = b["rows"].cpu()
            for e, r in readers.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    L[e][rows] = r(f, b)["logits"].float().cpu()
    df = RC.load_data(EXCL)["val"][0]
    idx = pd.Series(np.arange(len(M)), index=M.ann_id.values).loc[df.ann_id.values].values
    check = {}
    for e in (1, 10):
        assert not torch.isnan(L[e]).any(), e
        saved = np.load(os.path.join(probe, f"scores_val_ep{e}.npy")).astype(np.float64).ravel()
        new = L[e][idx].numpy().astype(np.float64).ravel()
        check[f"ep{e}"] = float(np.corrcoef(saved, new)[0, 1])
        assert check[f"ep{e}"] > .999, check
    for e, v in L.items():
        out = os.path.join(root, "preds", "r2", f"ep{e}")
        os.makedirs(out, exist_ok=True)
        np.savez(os.path.join(out, "probs_val.npz"), ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    CK.atomic_json(dict(step=step, n_rows=len(M), n_images=int(M.image_id.nunique()), reproduction_pearson=check, time=time.strftime("%F %T")), os.path.join(root, "infer_ok.json"))
    print(json.dumps(check), flush=True)
    return 0


def official(cfg, step):
    """Official Val AP of one step: inference (one GPU), the unmodified official evaluator for reader epochs 1-10 (CPU, in parallel), epoch selection. Idempotent."""
    root = official_root(cfg, step)
    os.makedirs(root, exist_ok=True)
    if os.path.exists(os.path.join(root, "DONE.json")):
        return 0
    py = [sys.executable, "-u", "-m", "src.crop_aware_pretrain.evals"]
    if not os.path.exists(os.path.join(root, "infer_ok.json")):
        rc = subprocess.call(py + ["official-infer", "--step", str(step)] + [x for ov in OVERRIDES for x in ("--set", ov)], cwd=CF.ROOT)
        if rc:
            return rc
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=OFFICIAL_PATH, PYTHONUNBUFFERED="1")
    procs = {}
    for e in range(1, 11):
        out = os.path.join(root, "eval_val", "r2", f"ep{e}")
        if os.path.exists(os.path.join(out, "official_results.json")):
            continue
        os.makedirs(out, exist_ok=True)
        procs[e] = subprocess.Popen([EPY, "-u", "-m", "src.pacoattr.official_eval", "--split", "val", "--pred", os.path.join(root, "preds", "r2", f"ep{e}", "probs_val.npz"), "--out", out], cwd=CF.ROOT, env=env,
                                    stdout=open(os.path.join(out, "log.txt"), "w"), stderr=subprocess.STDOUT)
    bad = {e: p.wait() for e, p in procs.items()}
    if any(bad.values()):
        print("official_eval failed", bad, flush=True)
        return 1
    rc = subprocess.call([sys.executable, "-m", "src.siglip_pm.select_epoch", root], cwd=CF.ROOT, stdout=subprocess.DEVNULL)
    if rc:
        return rc
    sel = json.load(open(os.path.join(root, "selection.json")))["r2"]
    CK.atomic_json(dict(step=step, epoch=sel["epoch"], val_obj=sel["val_obj"], val_part=sel["val_part"], time=time.strftime("%F %T")), os.path.join(root, "DONE.json"))
    print(json.dumps(dict(step=step, epoch=sel["epoch"], obj=sel["val_obj"], part=sel["val_part"])), flush=True)
    return 0


OFFICIAL_COLS = [("Overall", None), ("Object", None), ("Part", None), ("Obj Color", ("obj-attrs", "color")), ("Obj Pattern", ("obj-attrs", "pattern_marking")), ("Obj Material", ("obj-attrs", "material")),
                 ("Part Color", ("obj-part-attrs", "color")), ("Part Pattern", ("obj-part-attrs", "pattern_marking")), ("Part Material", ("obj-part-attrs", "material"))]


def _official_row(h):
    obj, part = h["obj-attr-AP"], h["obj-attr-part-AP"]
    row = {"Overall": (obj + part) / 2, "Object": obj, "Part": part}
    for name, key in OFFICIAL_COLS[3:]:
        row[name] = h[key[0]][key[1]]
    return {k: 100 * v if v <= 1 else v for k, v in row.items()}


def official_summary(cfg):
    D = CF.dirs(cfg)
    # Different checkpoints can finish CPU scoring at the same time. Serialize
    # the shared summary from the read of completed checkpoints to publication.
    with open(os.path.join(D["evals"], "official_summary.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _write_official_summary(cfg, D)


def _write_official_summary(cfg, D):
    rows = {}
    c0 = cfg["eval"]["baseline_c0_official"]
    sel = json.load(open(os.path.join(c0, "selection.json")))["C0_r2"]
    rows["C0 Crop Oracle"] = (sel["epoch"], _official_row(json.load(open(os.path.join(c0, "eval_val", "C0_r2", f"ep{sel['epoch']}", "official_results.json")))["hierarchical"]))
    steps = []
    for s in sorted(int(x.split("_")[1]) for x in os.listdir(D["evals"]) if x.startswith("step_")):
        f = os.path.join(official_root(cfg, s), "DONE.json")
        if os.path.exists(f):
            e = json.load(open(f))["epoch"]
            rows[f"step {s}"] = (e, _official_row(json.load(open(os.path.join(official_root(cfg, s), "eval_val", "r2", f"ep{e}", "official_results.json")))["hierarchical"]))
            steps.append(s)
    names = [f"step {s}" for s in steps] + ["C0 Crop Oracle"]
    cols = [c[0] for c in OFFICIAL_COLS]
    with open(os.path.join(D["evals"], "official_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "reader_epoch"] + cols)
        for n in names:
            w.writerow([n, rows[n][0]] + [f"{rows[n][1][c]:.3f}" for c in cols])
    L = ["# dinov3_crop_aware_pretrain: official PACO attribute AP, Val (GT box + category oracle; reader epoch = max (Object + Part) / 2 on official Val)", "",
         "| name | reader ep | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for n in names:
        L.append(f"| {n} | {rows[n][0]} | " + " | ".join(f"{rows[n][1][c]:.2f}" for c in cols) + " |")
    if "step 0" in rows and len(steps) > 1:
        b, c0r = rows["step 0"][1], rows["C0 Crop Oracle"][1]
        L += ["", "## Change vs step 0 (official Val) and share of the C0 - step 0 gap closed", "", "| step | Overall | Object | Part | Overall gap closed % |", "|---|---|---|---|---|"]
        for s in steps[1:]:
            r = rows[f"step {s}"][1]
            L.append(f"| {s} | {r['Overall'] - b['Overall']:+.2f} | {r['Object'] - b['Object']:+.2f} | {r['Part'] - b['Part']:+.2f} | {100 * (r['Overall'] - b['Overall']) / (c0r['Overall'] - b['Overall']):.1f} |")
    open(os.path.join(D["evals"], "official_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return 0


def diag(cfg, step):
    """COCO dev feature diagnostic of one exported EMA encoder (= the one train.py used to run in-process) -> appended to logs/diag.jsonl."""
    import pandas as pd
    import torch
    from src.config import COCO_TRAIN, FEAT
    from src.g1x.data import CocoDS
    from src.g1x.distill import dev_diag
    from src.msd.pretrain import eval_exclusions
    from src.viz.c0_pca import BACKBONES, load_encoder
    dev = torch.device("cuda")
    enc, enc0 = load_encoder(cfg["backbone"], dev), load_encoder(cfg["backbone"], dev)
    enc.load_state_dict(torch.load(os.path.join(step_dir(cfg, step), "enc_ema.pt"), map_location="cpu")["enc"])
    rois = pd.read_parquet(os.path.join(FEAT, "dev_rois.parquet"))
    rois = rois[~rois.image_id.isin(eval_exclusions()["raev2_dev"])]
    ids = sorted(int(i) for i in rois.image_id.unique())
    S, P = BACKBONES[cfg["backbone"]]["S"], BACKBONES[cfg["backbone"]]["P"]
    r = dev_diag(enc, enc0, None, CocoDS(rois, COCO_TRAIN, S, P), [ids[i:i + 8] for i in range(0, len(ids), 8)], dev)
    with open(os.path.join(CF.dirs(cfg)["logs"], "diag.jsonl"), "a") as f:
        f.write(json.dumps(dict(step=step, model="ema", **r, time=time.strftime("%F %T"))) + "\n")
    print(json.dumps(dict(step=step, roi_pool_cos=r["roi_pool_cos"])), flush=True)
    return 0


def run(cfg, step, limit_epochs=0):
    d = step_dir(cfg, step)
    if step > 0 and cfg["eval"].get("diag", True):                                # non-fatal: a failed diagnostic must not cost the PACO evaluation
        subprocess.call([sys.executable, "-u", "-m", "src.crop_aware_pretrain.evals", "diag", "--step", str(step)] + [x for ov in OVERRIDES for x in ("--set", ov)], cwd=CF.ROOT)
    out = os.path.join(d, "probe")
    if os.path.exists(out):                     # an earlier attempt that was interrupted: the probe has no resume, restart it
        shutil.rmtree(out)
    job = dict(PROBE, name=f"crop_aware_step{step}", encoder_ckpt=os.path.join(d, "enc_ema.pt"))
    if limit_epochs:
        job["epochs"] = limit_epochs
    path = os.path.join(d, "probe.yaml")
    yaml.safe_dump(job, open(path, "w"), sort_keys=False)
    cmd = [sys.executable, "-u", "-m", "src.g1x.probe", "--job", path, "--out", out, "--exclude", cfg["eval"]["exclude"], "--workers", str(cfg["eval"]["workers"])]
    p = subprocess.Popen(cmd, cwd=CF.ROOT)
    signal.signal(signal.SIGTERM, lambda *_: p.terminate())
    rc = p.wait()
    if rc != 0:
        return rc
    CK.atomic_json(dict(step=step, done=time.strftime("%F %T")), os.path.join(d, "probe_ok.json"))
    orc = official(cfg, step)                   # official AP is the reported metric; a failure is recorded and can be retried with `official --step`
    if orc:
        CK.atomic_json(dict(step=step, returncode=orc, time=time.strftime("%F %T")), os.path.join(official_root(cfg, step), "FAILED.json"))
    else:
        official_summary(cfg)
    return 0


COLS_M2 = [("M2 Overall", "m2", "all", "overall"), ("M2 Object", "m2", "all", "object"), ("M2 Part", "m2", "all", "part"),
           ("M2 Color", "m2", "all", "color"), ("M2 Material", "m2", "all", "material"), ("M2 Pattern", "m2", "all", "pattern_marking")]
COLS_M0 = [("M0 Overall", "m0", "all", "overall"), ("M0 Obj D5-16", "m0", "D5-16", "object"), ("M0 Part D1-4", "m0", "D1-4", "part")]
CACHE = "metrics.npz"


def _metrics(BB, S, path):
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    out = {}
    for b in BB:
        for kind in ("m0", "m2"):
            try:
                r = getattr(BB[b], kind)(S)
            except ValueError:
                r = None
            for g, v in (r or {}).items():
                out[f"{b}|{kind}|{g}"] = np.asarray(v, np.float32)
    np.savez(path + ".tmp.npz", **out)
    os.replace(path + ".tmp.npz", path)
    return out


def report(cfg, base_only=False):
    import src.readout.common as RC
    from src.ivc.common import DBINS, EXCL
    from src.ivc.compare_bb import pick
    from src.readout.evaluate import Boot, summarize
    D = CF.dirs(cfg)
    data = RC.load_data(EXCL)
    dva, Yva = data["val"]
    ytr = RC.ytr_by_level(data)
    nb = int(cfg["eval"]["bootstrap"])
    BB = {"all": Boot(dva, Yva, ytr, nb=nb), **{nm: Boot(dva, Yva, ytr, nb=nb, mask=(dva.n_in.values >= lo) & (dva.n_in.values <= hi)) for nm, lo, hi in DBINS}}
    rows, M = [], {}
    def load(name, path):
        ep, dm0, S = pick(os.path.join(path, "r2"))
        assert S.shape == (len(dva), 59), S.shape
        os.makedirs(os.path.join(path, "_ca_metrics"), exist_ok=True) if name.startswith("step") else None
        cache = os.path.join(path, "_ca_metrics", CACHE) if name.startswith("step") else os.path.join(D["evals"], f"{name}.{CACHE}")
        M[name] = _metrics(BB, S, cache)
        return dict(name=name, reader_epoch=ep, dev_M0=round(dm0, 3))

    meta = {}
    meta["FrozenFull(existing)"] = load("FrozenFull(existing)", cfg["eval"]["baseline_full"])
    meta["C0(existing)"] = load("C0(existing)", cfg["eval"]["baseline_c0"])
    steps = []
    for s in sorted(int(x.split("_")[1]) for x in os.listdir(D["evals"]) if x.startswith("step_")):
        d = step_dir(cfg, s)
        if os.path.exists(os.path.join(d, "probe_ok.json")):
            nm = f"step{s}"
            meta[nm] = load(nm, os.path.join(d, "probe"))
            steps.append(s)
    def pt(nm, b, k, g):
        v = M[nm].get(f"{b}|{k}|{g}")
        return float(v[0]) if v is not None and np.isfinite(v[0]) else float("nan")
    names = ["FrozenFull(existing)", "C0(existing)"] + [f"step{s}" for s in steps]
    cols = COLS_M2 + COLS_M0
    with open(os.path.join(D["evals"], "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "reader_epoch", "dev_M0"] + [c[0] for c in cols])
        for nm in names:
            w.writerow([nm, meta[nm]["reader_epoch"], meta[nm]["dev_M0"]] + [f"{pt(nm, b, k, g):.3f}" for _, k, b, g in cols])
    base = "step0" if "step0" in M else "FrozenFull(existing)"
    L = ["# dinov3_crop_aware_pretrain: PACO val R2 (dev-selected epoch; metric: project M2/M0 as named in the spec, NOT official attribute AP)", "",
         "| name | reader ep | " + " | ".join(c[0] for c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for nm in names:
        L.append(f"| {nm} | {meta[nm]['reader_epoch']} | " + " | ".join(f"{pt(nm, b, k, g):.2f}" for _, k, b, g in cols) + " |")
    L += ["", f"## Paired image bootstrap vs {base} (95% CI, B={nb})", "", "| comparison | " + " | ".join(c[0] for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    gaps = []
    for nm in [x for x in names if x != base]:
        cells = []
        for _, k, b, g in cols:
            x, y = M[nm].get(f"{b}|{k}|{g}"), M[base].get(f"{b}|{k}|{g}")
            if x is None or y is None:
                cells.append("n/a")
                continue
            r = summarize(x, y)
            star = "*" if r["lo"] > 0 or r["hi"] < 0 else ""
            cells.append(f"{r['point']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]{star}")
        L.append(f"| {nm} − {base} | " + " | ".join(cells) + " |")
    if "C0(existing)" in M and base in M:
        L += ["", "## Full-to-crop gap closed (C0 minus Full; 100% = trained encoder matches C0)", "",
              "| step | gap M2 Overall | gap M0 Overall | gap M0 Obj D5-16 | M2 Overall closed % |", "|---|---|---|---|---|"]
        f0 = pt(base, "all", "m2", "overall")
        c0 = pt("C0(existing)", "all", "m2", "overall")
        for s in steps:
            nm = f"step{s}"
            d = pt("C0(existing)", "all", "m2", "overall") - pt(nm, "all", "m2", "overall")
            closed = 100 * (pt(nm, "all", "m2", "overall") - f0) / (c0 - f0) if c0 != f0 else float("nan")
            gaps.append(dict(step=s, gap_m2=d, closed_pct=closed))
            L.append(f"| {s} | {d:+.2f} | {pt('C0(existing)', 'all', 'm0', 'overall') - pt(nm, 'all', 'm0', 'overall'):+.2f} | "
                     f"{pt('C0(existing)', 'D5-16', 'm0', 'object') - pt(nm, 'D5-16', 'm0', 'object'):+.2f} | {closed:.1f} |")
    open(os.path.join(D["evals"], "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    decide(cfg, steps, {s: pt(f"step{s}", "all", "m2", "overall") for s in steps}, D)
    return 0


def decide(cfg, steps, m2, D):
    """Fixed stop rule (val M2 Overall). Written once the gate step has been evaluated; the supervisor halts training on 'halt'."""
    rule = cfg["eval"]["stop_rule"]
    gate = int(rule["gate_step"])
    f = os.path.join(D["control"], "stop_decision.json")
    if 0 not in steps or gate not in steps:
        return
    base = m2[0]
    used = [s for s in steps if 0 < s <= gate]
    delta = {s: m2[s] - base for s in used}
    best = max(delta.values()) if delta else float("nan")
    last3 = [delta[s] for s in used[-3:]]
    reasons = []
    if best < float(rule["min_best_gain"]):
        reasons.append(f"best delta over steps <= {gate} is {best:+.2f} < {rule['min_best_gain']}")
    if len(last3) == 3 and all(x < -float(rule["worse_tolerance"]) for x in last3):
        reasons.append(f"last three evals all worse than step 0 by more than {rule['worse_tolerance']}: {[round(x, 2) for x in last3]}")
    CK.atomic_json(dict(decision="halt" if reasons else "continue", reasons=reasons, gate_step=gate, val_m2_overall=m2, delta_vs_step0=delta, time=time.strftime("%F %T")), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report", "diag", "official", "official-infer", "official-summary"))
    ap.add_argument("--all", action="store_true", help="official: every step whose probe finished and has no official result yet")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0, help="smoke override")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = CF.load(a.config, a.set)
    OVERRIDES[:] = a.set
    os.environ.setdefault("OMP_NUM_THREADS", "4")   # probe subprocesses inherit this: without it each probe spins dozens of OpenMP threads and starves the training process
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.nice(19)                                 # evals (probe data loading, official evaluator, children inherit this) must not take CPU time from a training process sharing the machine
    if a.cmd == "diag":
        sys.exit(diag(cfg, a.step))
    if a.cmd == "official-infer":
        sys.exit(official_infer(cfg, a.step))
    if a.cmd == "official-summary":
        sys.exit(official_summary(cfg))
    if a.cmd == "official":
        if a.all:
            from concurrent.futures import ThreadPoolExecutor
            E = CF.dirs(cfg)["evals"]
            todo = [s for s in sorted(int(x.split("_")[1]) for x in os.listdir(E) if x.startswith("step_"))
                    if os.path.exists(os.path.join(step_dir(cfg, s), "probe_ok.json")) and not os.path.exists(os.path.join(step_dir(cfg, s), "RUNNING"))]
            print("official steps", todo, flush=True)
            with ThreadPoolExecutor(2) as pool:
                rcs = list(pool.map(lambda st: official(cfg, st), todo))
            print("return codes", dict(zip(todo, rcs)), flush=True)
            official_summary(cfg)
            sys.exit(max(rcs) if rcs else 0)
        rc = official(cfg, a.step)
        if not rc:
            official_summary(cfg)
        sys.exit(rc)
    if a.cmd == "run":
        rc = run(cfg, a.step, a.epochs)
        if rc == 0 and cfg["eval"].get("m2_report", True):
            rc = report(cfg)
        sys.exit(rc)
    sys.exit(report(cfg))


if __name__ == "__main__":
    main()
