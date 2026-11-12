"""Early G1U / G1R point-estimate report: official Val and Test AP as soon as they exist, without waiting for the queue's bootstrap and ROI-size stages.

Runs beside src.g1up.queue and never writes to its files. CPU only. Reads the epochs the queue froze on official Val (eval/selection_frozen.json); Test never changes them.
Writes <run>/eval/fast_ap/{fast_ap.json, tables.md, done.json or failed.json}.

  setsid nohup python -u -m src.g1up.fast_ap --run output/2026-11-09/g1_upscaled_teacher_s0 &
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path("/home/irteam/Crop_representation")
EPY = "/home/irteam/envs/paco_eval/bin/python"
BASELINE_TEST = ROOT / "output/2026-11-09/pixeldit_cond_p1p2_s0/official_test/eval_boot"
NEW = ("G1U_r2", "G1R_r2")
LINKED = ("Cosine_r2", "Hybrid_r2", "C0_r2")
LABEL = {"G1U_r2": "G1U", "G1R_r2": "G1R", "Cosine_r2": "Crop Cosine P384", "Hybrid_r2": "PixelDiT + Crop Cosine P384", "C0_r2": "C0 Crop Oracle"}
ORDER = ("Cosine_r2", "Hybrid_r2", "G1U_r2", "G1R_r2", "C0_r2")
COLUMNS = (("overall", "Overall"), ("object", "Object"), ("part", "Part"), ("pattern", "Pattern"), ("material", "Material"), ("obj_pattern", "Obj Pattern"), ("obj_material", "Obj Material"), ("part_pattern", "Part Pattern"), ("part_material", "Part Material"))
PAIRS = (("G1U_r2", "Hybrid_r2"), ("G1R_r2", "Hybrid_r2"), ("G1R_r2", "G1U_r2"), ("C0_r2", "G1U_r2"), ("C0_r2", "G1R_r2"))


def row(path):
    h = json.loads(Path(path).read_text())["hierarchical"]
    o, p = h["obj-attrs"], h["obj-part-attrs"]
    return dict(overall=(h["obj-attr-AP"] + h["obj-attr-part-AP"]) / 2, object=h["obj-attr-AP"], part=h["obj-attr-part-AP"], pattern=(o["pattern_marking"] + p["pattern_marking"]) / 2, material=(o["material"] + p["material"]) / 2, obj_pattern=o["pattern_marking"], obj_material=o["material"], part_pattern=p["pattern_marking"], part_material=p["material"])


def table(rows, selection):
    head = ["Model", "epoch"] + [c for _, c in COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join([LABEL[n], str(selection[n]["epoch"])] + [f"{rows[n][k]:.2f}" for k, _ in COLUMNS]) + " |" for n in ORDER]
    lines += ["", "| Contrast (point estimate) | | " + " | ".join(c for _, c in COLUMNS) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join([f"{LABEL[x]} - {LABEL[y]}", ""] + [f"{rows[x][k] - rows[y][k]:+.2f}" for k, _ in COLUMNS]) + " |" for x, y in PAIRS]
    return "\n".join(lines)


def queue_alive():
    return any("-m src.g1up.queue" in line and "bash -c" not in line for line in subprocess.check_output(["ps", "-eo", "args"], text=True).splitlines())


def wait(run, paths):
    while not all(Path(p).exists() for p in paths):
        for name in ("queue_failed.json", "queue_held.json"):
            if (run / name).exists():
                raise RuntimeError(f"queue stopped: {name}")
        failed = sorted(x.name for x in (run / "state").glob("*.failed.json"))
        if failed:
            raise RuntimeError(f"queue stage failed: {failed}")
        if not queue_alive():
            time.sleep(20)                       # the last file may land just as the queue moves on
            if not all(Path(p).exists() for p in paths):
                raise RuntimeError("queue process is gone")
            return
        time.sleep(10)


def evaluate(pred, out):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONPATH=f"/home/irteam/datasets/lvis_src/paco:/home/irteam/datasets/lvis_src/lvis-api:{ROOT}")
    subprocess.run([EPY, "-u", "-m", "src.pacoattr.official_eval", "--split", "test", "--pred", str(pred), "--out", str(out)], cwd=ROOT, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = ROOT / a.run
    root, out = run / "eval", run / "eval/fast_ap"
    out.mkdir(parents=True, exist_ok=True)
    try:
        frozen = root / "selection_frozen.json"
        wait(run, [frozen])
        sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
        selection = json.loads(frozen.read_text())
        val = {n: row(root / f"eval_val/{n}/ep{selection[n]['epoch']}/official_results.json") for n in ORDER}
        (out / "val_ap.json").write_text(json.dumps(dict(selection=selection, val=val), indent=1))
        print("val ready", flush=True)
        wait(run, [run / "state/test_infer_G1U.done.json", run / "state/test_infer_G1R.done.json"])
        preds = {n: root / f"preds/{n}/ep{selection[n]['epoch']}/probs_test.npz" for n in NEW}
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda n: evaluate(preds[n], out / "test" / n), NEW))
        test = {n: row(out / "test" / n / "official_results.json") for n in NEW}
        test.update({n: row(BASELINE_TEST / n / "test/official_results.json") for n in LINKED})
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == sha, "selection changed during Test evaluation"
        (out / "fast_ap.json").write_text(json.dumps(dict(selection=selection, selection_sha256=sha, val=val, test=test), indent=1))
        md = ["# G1U / G1R official AP (point estimates; bootstrap and ROI size follow separately)", "", "Epoch = max (Object + Part) / 2 on official Val, fixed before Test. Pattern and Material = mean of Object and Part. R2 reader, encoder seed 0.", "", "## Official Val", "", table(val, selection), "", "## Official Test", "", table(test, selection), ""]
        (out / "tables.md").write_text("\n".join(md))
        (out / "done.json").write_text(json.dumps(dict(status="done", time=time.strftime("%FT%TZ", time.gmtime())), indent=1))
        print("done", flush=True)
    except Exception:
        (out / "failed.json").write_text(json.dumps(dict(status="failed", traceback=traceback.format_exc()), indent=1))
        raise


if __name__ == "__main__":
    main()
