"""P1 / P2 on PACO official Test at the epochs fixed on official Val (official_val/selection.json), once. Results go to <run>/official_test. Selection is never revisited.

Waits for official_val/done.json, then: Test inference (P1, P2 R2 + Mean; Initial, Hybrid Mean, whose official Test predictions do not exist yet) -> official evaluator -> paired bootstrap B = 1000 seed 0 -> ROI size -> tables.
Existing Test predictions of the other baselines are read, never rewritten.

  python -m src.pdit.official_test queue --run output/2026-11-09/pixeldit_cond_p1p2_s0
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PY, EPY = "/home/irteam/envs/dinov2_ord_py310/bin/python", "/home/irteam/envs/paco_eval/bin/python"
ROOT = "/home/irteam/Crop_representation"
OVAL = Path(ROOT) / "output/2026-11-09/hcf_official_val"
PILOT = Path(ROOT) / "output/2026-11-08/hcf_pilot"
HIER = Path(ROOT) / "output/2026-11-09/hier_h0h1_s0/test_eval"
CAUSAL = Path(ROOT) / "output/2026-11-08/hcf_material_causal"
MODELS = ("Initial", "Cosine", "Hybrid", "P1", "P2", "C0")
EXISTING = {"Initial_r2": (OVAL, "r384_initial"), "Cosine_r2": (OVAL, "r384_cosine"), "Hybrid_r2": (OVAL, "r384_pixeldit_hybrid"), "C0_r2": (PILOT, "c0_r2"), "Cosine_linear": (HIER, "P384_linear"), "C0_linear": (HIER, "C0_linear")}


def infer(a):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    import src.readout.common as RC
    from src.hcf.model import load_hcf
    from src.hcf.probe import encode
    from src.pdit.bbdata import BBImageDS
    from src.pdit.official_infer import load_reader
    dev, t0, root = torch.device("cuda"), time.time(), Path(a.root)
    selection = json.loads((root.parent / "official_val/selection.json").read_text())
    probe = Path(a.run) / a.cond / "probe" if a.cond in ("P1", "P2") else CAUSAL / {"Initial": "r384_initial", "Hybrid": "r384_pixeldit_hybrid"}[a.cond] / "probe"
    model, ck = load_hcf(json.loads((probe / "config.json").read_text())["job"]["encoder_ckpt"], dev)
    assert ck["kind"] == "ema"
    specs = [("r2", "r2", probe / "r2fused"), ("linear", "r0", probe / "r0")] if a.cond in ("P1", "P2") else [("linear", "r0", probe / "r0")]
    M = pd.read_parquet(f"{ROOT}/output/2026-11-07/paco_oracle_attr/data/manifest_test.parquet")
    assert set(M.split.unique()) == {"test"} and M.image_id.nunique() == 9443
    ds = BBImageDS(M, np.zeros((len(M), 59), np.int8), False, "bil", 384, 16)
    readers = {key: (selection[f"{a.cond}_{key}"]["epoch"], load_reader(kind, path / f"ep{selection[f'{a.cond}_{key}']['epoch']}.pt", dev)) for key, kind, path in specs}
    L = {k: torch.full((len(M), 59), float("nan")) for k in readers}
    chunks = [list(range(i, min(len(ds), i + 12))) for i in range(0, len(ds), 12)]
    with torch.no_grad():
        for b in DataLoader(ds, sampler=chunks, batch_size=1, num_workers=8, collate_fn=RC.collate_one, pin_memory=True):
            b = RC.to_dev(b, dev)
            rows = b["rows"].cpu()
            f = {12: encode(model, b, ["Q"])["Q"]}
            for k, (_, r) in readers.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    L[k][rows] = r(f, b)["logits"].float().cpu()
    for key, v in L.items():
        assert not torch.isnan(v).any(), key
        out = root / "preds" / f"{a.cond}_{key}" / f"ep{readers[key][0]}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "probs_test.npz", ann_id=M.ann_id.values, image_id=M.image_id.values, category_id=M.category_id.values, logits=v.numpy().astype(np.float32), probs=torch.sigmoid(v).numpy().astype(np.float32))
    print(json.dumps(dict(cond=a.cond, n_rows=len(M), seconds=round(time.time() - t0))), flush=True)


def report(a):
    import numpy as np
    import pandas as pd

    from src.pdit.official_report import COLUMNS, LABEL, PAIRS, boot_values, md, point
    root = Path(a.run) / "official_test"
    out = root / "results"
    selection = json.loads((Path(a.run) / "official_val/selection.json").read_text())
    lines = ["# PACO official Test (GT box + GT category oracle attribute AP)", "", "Test evaluated once at the epochs fixed on official Val (max (Object + Part) / 2, ties earlier); nothing is reselected. Paired image bootstrap B = 1000, seed 0. Overall = (Object + Part) / 2. Encoder seed 0 only.", ""]
    for key, name in (("r2", "R2 Cross-attention Reader"), ("linear", "Mean + Linear Probe")):
        boot = json.loads((root / f"boot_test_{key}.json").read_text())
        table = pd.DataFrame([dict(Model=LABEL[m], epoch=selection[f"{m}_{key}"]["epoch"], **{c: float(point(boot["points"][f"{m}_{key}"], c)) for c in COLUMNS}) for m in MODELS])
        rows, csv = [], []
        for x, y in PAIRS:
            X, Y = boot["boot"][f"{x}_{key}"], boot["boot"][f"{y}_{key}"]
            row = dict(Contrast=f"{x} - {y}")
            for c in COLUMNS:
                d = float(point(boot["points"][f"{x}_{key}"], c) - point(boot["points"][f"{y}_{key}"], c))
                lo, hi = np.percentile(boot_values(X, c) - boot_values(Y, c), [2.5, 97.5])
                row[c] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                csv.append(dict(reader=key, contrast=f"{x} - {y}", metric=c, delta=d, lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0)))
            rows.append(row)
        table.to_csv(out / f"test_ap_{key}.csv", index=False)
        pd.DataFrame(csv).to_csv(out / f"test_contrasts_{key}.csv", index=False)
        size = pd.concat([pd.read_csv(out / f"roi_size_{m}_{key}.csv") for m in MODELS], ignore_index=True)
        size.to_csv(out / f"test_roi_size_{key}.csv", index=False)
        lines += [f"## {name}", "", md(table), "", "### Contrasts (95% CI)", "", md(pd.DataFrame(rows)), "", "### ROI size (GT-box area on the 384 canvas, point estimates)", ""]
        for level in ("Object", "Part"):
            for col in ("Pattern", "Material"):
                p = size[size.level == level].pivot(index="size_bin", columns="model", values=col).reindex(columns=[f"{m}_{key}" for m in MODELS]).reindex(["tiny_lt4_tokens", "small_4to16_tokens", "medium_16to64_tokens", "large_ge64_tokens"])
                count = size[(size.level == level) & (size.model == f"P1_{key}")].set_index("size_bin").roi_count
                t = p.reset_index()
                t.insert(1, "ROIs", [int(count[s]) for s in p.index])
                t.columns = ["Size", "ROIs"] + list(MODELS)
                lines += [f"**{level} {col}**", "", md(t), ""]
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def queue(a):
    run = Path(ROOT) / a.run
    root = run / "official_test"
    for d in ("logs", "state", "results", "preds"):
        (root / d).mkdir(parents=True, exist_ok=True)
    lock = open(root / "queue.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("official_test queue already running")

    def stage(name, cmd_args, gpu=None, official=False):
        done = root / "state" / f"{name}.done.json"
        if done.exists():
            return
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        if official:
            env["PYTHONPATH"] = f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}"
        started = time.time()
        with (root / "logs" / f"{name}.log").open("a") as log:
            rc = subprocess.run([EPY if official else PY, "-u", "-m"] + list(map(str, cmd_args)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        record = dict(stage=name, gpu=gpu, returncode=rc, seconds=time.time() - started)
        (done if rc == 0 else root / "state" / f"{name}.failed.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(record), flush=True)
        if rc:
            raise RuntimeError(name)

    def parallel(jobs, n):
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda f: f(), jobs))

    try:
        val = run / "official_val"
        while not (val / "done.json").exists():
            if (val / "queue_failed.json").exists():
                raise RuntimeError("official Val queue failed")
            time.sleep(30)
        selection = json.loads((val / "selection.json").read_text())
        names = [f"{m}_{k}" for m in MODELS for k in ("r2", "linear")]
        for name, (base, old) in EXISTING.items():
            path = base / "preds" / old / f"ep{selection[name]['epoch']}/probs_test.npz"
            assert path.exists(), (name, path)

        def pred(n):
            e = selection[n]["epoch"]
            if n in EXISTING:
                base, old = EXISTING[n]
                return base / "preds" / old / f"ep{e}/probs_test.npz"
            return root / "preds" / n / f"ep{e}/probs_test.npz"

        parallel([lambda c=c, g=g: stage(f"infer_{c}", ["src.pdit.official_test", "infer", "--root", root, "--run", run, "--cond", c], g) for c, g in (("P1", 0), ("P2", 1))], 2)
        parallel([lambda c=c, g=g: stage(f"infer_{c}", ["src.pdit.official_test", "infer", "--root", root, "--run", run, "--cond", c], g) for c, g in (("Initial", 0), ("Hybrid", 1))], 2)
        parallel([lambda n=n: stage(f"eval_{n}", ["src.pacoattr.official_eval", "--split", "test", "--pred", pred(n), "--out", root / f"eval_boot/{n}/test", "--boot"], official=True) for n in names], 12)
        for key in ("r2", "linear"):
            stage(f"bootstrap_{key}", ["src.g1up.support", "boot", "--root", root, "--names", ",".join(f"{m}_{key}" for m in MODELS), "--split", "test", "--out", root / f"boot_test_{key}.json"], official=True)
        parallel([lambda n=n: stage(f"size_{n}", ["src.g1up.support", "size", "--split", "test", "--pred", pred(n), "--model", n, "--out", root / f"results/roi_size_{n}.csv"], official=True) for n in names], 12)
        stage("report", ["src.pdit.official_test", "report", "--run", run])
        (root / "done.json").write_text(json.dumps(dict(status="done", completed_utc=time.strftime("%FT%TZ", time.gmtime())), indent=1))
    except Exception:
        (root / "queue_failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc()), indent=1))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("queue", "infer", "report"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--root")
    ap.add_argument("--cond")
    a = ap.parse_args()
    dict(queue=queue, infer=infer, report=report)[a.cmd](a)


if __name__ == "__main__":
    main()
