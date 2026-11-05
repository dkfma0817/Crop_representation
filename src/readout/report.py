"""Queue report: hypothesis -> training -> direct check -> downstream -> verdict, for every experiment (PACO VAL only).

Primary: val M2 Overall at the LAST epoch vs the queue baseline ctrl_f336/ml_r2 (Multi-layer + R2, blocks 4/8/12),
paired image bootstrap (1,000, seed 0). Secondary: object, part, color, material, pattern_marking, transparency.
Object 5-16 is not reported (official M2 on val is NaN; never imputed). Hypothesis check and downstream are separate.
Downstream success (pre-registered): dOverall CI lo > 0, AND not concentrated in one attribute type (a single type
< 70% of the summed positive type gains), AND no major degradation (no secondary metric with d <= -1.0 and CI hi < 0).
Run valid: done.json, all epochs logged, finite train/val losses.

  python -m src.readout.report --run output/<date>/readout_queue
"""
import argparse
import json
import os

import numpy as np

import src.readout.common as RC
from src.readout import d0_binding as D0
from src.readout.evaluate import GROUPS, Boot, summarize

BASE = ("ctrl_f336", "ml_r2")
TYPES = ("color", "pattern_marking", "material", "transparency")
EXPS = {
    "E05": dict(title="ROI Binding Supervision", readers=["e05_bind"], main="e05_bind",
                hyp="A ranking loss between ROIs of the same image (explicit positive vs explicit negative) teaches the shared-feature "
                    "reader to attach an attribute to the right ROI (less leakage from same-image neighbours).",
                train="Baseline ml_r2 + lambda 0.5 x softplus(s_j,a - s_i,a) over same-image, same-level pairs; BCE unchanged. "
                      "Control = ml_r2 (same batches, so the same pairs are seen; BCE only)."),
    "E08": dict(title="Weight-sharing Repeated Readout", readers=["e08_x1", "e08_x3", "e08_ffn3"], main="e08_x3",
                hyp="Reading the same shared features again with the same cross-attention finds information a single read missed.",
                train="One cross block: x1 reads once; x3 reads 3 times with shared weights; ffn3 reads once + 2 extra passes of the same "
                      "block's FFN (same params and extra compute, no re-reading). Loss on the final output only; every step recorded."),
    "E09": dict(title="Query-conditioned Adaptive Sampling", readers=["e09_ads", "e09_fix"], main="e09_ads",
                hyp="With the same number of feature samples (36), choosing WHERE to sample from the query improves discrimination.",
                train="ml_r2 + 36 bilinear samples of the same fused F336 map per block (box coords in [-0.25, 1.25]); ads predicts the "
                      "points from the query (initialised on the grid), fix keeps the grid. No new pixels, no crop encoding."),
    "E11": dict(title="Local/Context Gated Readout", readers=["e11_concat", "e11_gate", "e11_ctx"], main="e11_gate",
                hyp="Reading local and context evidence separately and gating them uses context better than one joint read of the same tokens.",
                train="concat (primary control): one stream, 2 blocks over [ROI tokens ; all tokens]; gate: local stream (ROI tokens) and "
                      "context stream (all tokens) + learnable per-channel gate; ctx: context-only ablation. Same tokens, same output dim."),
    "E12": dict(title="Scale Decision Consistency", readers=["e12_cons"], main="e12_cons",
                hyp="Training the F336 reader to agree with the decisions of a 672-encoded view of the SAME 336 image (no new pixels) "
                    "reduces the encoding-scale discrimination gap; inference uses F336 only.",
                train="Student = ml_r2 on F336, BCE + 1.0 x BCE(student, sigmoid(teacher)) on all 59 attributes of the same real ROI. "
                      "Teacher = ctrl_f672/f672_ml_r2 (336 canvas upsampled x2 -> DINOv2 672, same reader). Control = ml_r2 (BCE only)."),
}


def f(d, nd=2):
    if d is None or not np.isfinite(d.get("point", np.nan)):
        return "n/a"
    if not np.isfinite(d.get("lo", np.nan)):
        return f"{d['point']:.{nd}f}"
    return f"{d['point']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def p(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


class Run:
    def __init__(self, run, data):
        self.run, self.data = run, data
        self.dva, self.Yva = data["val"]
        self.boot = Boot(self.dva, self.Yva, RC.ytr_by_level(data))
        self._m2 = {}

    def job_status(self, job):
        d = os.path.join(self.run, job)
        if os.path.exists(os.path.join(d, "done.json")):
            return "done", json.load(open(os.path.join(d, "done.json")))
        if os.path.exists(os.path.join(d, "failed.json")):
            return "failed", json.load(open(os.path.join(d, "failed.json")))
        return ("missing" if not os.path.isdir(d) else "incomplete"), {}

    def curves(self, job, rd):
        fn = os.path.join(self.run, job, rd, "curves.jsonl")
        return [json.loads(l) for l in open(fn)] if os.path.exists(fn) else []

    def scores(self, job, rd):
        cv = self.curves(job, rd)
        if not cv:
            return None
        return np.load(os.path.join(self.run, job, rd, f"scores_val_ep{cv[-1]['epoch']}.npy")).astype(np.float32)

    def m2(self, key, S=None):
        if key not in self._m2:
            if S is None:
                S = self.scores(*key)
            self._m2[key] = None if S is None else self.boot.m2(S)
        return self._m2[key]

    def extra(self, job):
        fn = os.path.join(self.run, job, "extra_val.npz")
        return dict(np.load(fn)) if os.path.exists(fn) else {}

    def validity(self, job, rd, epochs=10):
        st, info = self.job_status(job)
        cv = self.curves(job, rd)
        finite = all(np.isfinite(r["val"]["loss"]) and np.isfinite(r.get("train_running_loss", 0.0)) for r in cv)
        ok = st == "done" and len(cv) == epochs + 1 and finite
        fit = {}
        fn = os.path.join(self.run, job, "train_fit.json")
        if os.path.exists(fn):
            fit = json.load(open(fn)).get(rd, {})
        return dict(valid=bool(ok), status=st, epochs_logged=max(0, len(cv) - 1), finite=finite, info=info,
                    train_loss=[round(r.get("train_running_loss", np.nan), 4) for r in cv[1:]],
                    val_loss=[round(r["val"]["loss"], 4) for r in cv],
                    val_M2=[round(r["val"]["M2"]["overall"], 2) for r in cv],
                    best_val_epoch=int(np.nanargmax([r["val"]["M2"]["overall"] for r in cv])) if cv else None,
                    train_subset_loss=fit.get("train_subset_loss"), train_subset_M2=(fit.get("train_subset_M2") or {}).get("overall"))


def downstream(R, key):
    mb, mm = R.m2(BASE), R.m2(key)
    if mb is None or mm is None:
        return None
    d = {g: summarize(mm[g], mb[g]) for g in GROUPS}
    pos = {t: max(0.0, d[t]["point"]) for t in TYPES}
    tot = sum(pos.values())
    top = max(pos, key=pos.get) if tot > 0 else None
    concentrated = bool(tot > 0 and pos[top] / tot >= 0.7)
    degraded = [g for g in GROUPS[1:] if d[g]["point"] <= -1.0 and d[g]["hi"] < 0]
    improved = bool(d["overall"]["lo"] > 0)
    return dict(abs={g: float(mm[g][0]) for g in GROUPS}, base={g: float(mb[g][0]) for g in GROUPS}, delta=d,
                overall_improved=improved, concentrated=concentrated, top_type=top,
                top_share=(pos[top] / tot if tot > 0 else None), major_degradation=degraded,
                downstream_success=bool(improved and not concentrated and not degraded))


def m2_diff(R, a, b, g="overall"):
    ma, mb = R.m2(a), R.m2(b)
    return None if ma is None or mb is None else summarize(ma[g], mb[g])


# ------------------------------------------------------------------------------------------------ direct checks
def check_E05(R):
    job = "E05"
    S_e, S_b = R.scores(job, "e05_bind"), R.scores(*BASE)
    if S_e is None or S_b is None:
        return dict(ability=None, note="missing scores")
    nin = np.load(RC.NIN_SG.format("val"))
    pf = os.path.join(R.run, "D0", "pairs.parquet")
    import pandas as pd
    if os.path.exists(pf):
        P = pd.read_parquet(pf)
        X = P[[c for c in P.columns if c.startswith("x") and c[1:].isdigit()]].to_numpy()
    else:
        P, X = D0.build_pairs(R.dva, R.Yva, nin)
    Wp = R.boot.W[:, P.i.values]
    stats = {"e05_bind": D0.pair_stats(S_e, P, X), "ml_r2": D0.pair_stats(S_b, P, X)}
    res = D0.analyse(stats, P, Wp, base="e05_bind", ref="ml_r2")
    al = res["all"]["e05_bind-ml_r2"]
    dP_e = D0.boot_mean(stats["e05_bind"]["dP"], Wp, np.ones(len(P), bool))
    dP_b = D0.boot_mean(stats["ml_r2"]["dP"], Wp, np.ones(len(P), bool))
    dP = summarize(dP_e, dP_b)
    ability = bool(al["PA_w_diff"]["lo"] > 0 and al["BD_diff"]["hi"] < 0)
    rows = [("paired accuracy PA_w (e05 - base)", f(al["PA_w_diff"])), ("cross-image PA_x (e05 - base)", f(al["PA_x_diff"])),
            ("binding deficit BD (e05 - base)", f(al["BD_diff"])), ("ROI-swap prediction change dP (e05 - base)", f(dP, 4))]
    for sn in ("obj_small", "obj_large", "part", "same_cat", "overlap"):
        rows.append((f"PA_w diff [{sn}, n={res[sn]['n_pairs']}]", f(res[sn]["e05_bind-ml_r2"]["PA_w_diff"])))
    return dict(ability=ability, rule="PA_w(e05) - PA_w(base) CI lo > 0 AND BD(e05) - BD(base) CI hi < 0", rows=rows)


def check_E08(R):
    ex = R.extra("E08")
    rows, out = [], {}
    for rd in ("e08_x1", "e08_x3", "e08_ffn3"):
        k = f"{rd}__step_logits"
        if k in ex:
            for t in range(ex[k].shape[1]):
                m = R.m2(("E08", f"{rd}@step{t + 1}"), ex[k][:, t])
                rows.append((f"{rd} step {t + 1} val M2 Overall", p(float(m["overall"][0]))))
    a, b, c = ("E08", "e08_x3"), ("E08", "e08_x1"), ("E08", "e08_ffn3")
    d1, d2 = m2_diff(R, a, b), m2_diff(R, a, c)
    rows += [("x3 - x1 (Overall)", f(d1)), ("x3 - ffn3 (Overall)", f(d2)),
             ("x3 - x1 (Part)", f(m2_diff(R, a, b, "part"))), ("x3 - ffn3 (Part)", f(m2_diff(R, a, c, "part")))]
    ability = None if d1 is None or d2 is None else bool(d1["lo"] > 0 and d2["lo"] > 0)
    return dict(ability=ability, rule="x3 - x1 CI lo > 0 AND x3 - ffn3 CI lo > 0 (val M2 Overall)", rows=rows)


def _masks_for_val(dva):
    from pycocotools import mask as MU
    ann = json.load(open(os.path.join(RC.os.environ.get("PACO_DIR", "/home/irteam/datasets/paco"), "paco_lvis_v1_val.json")))
    img = {i["id"]: (i["height"], i["width"]) for i in ann["images"]}
    by = {a["id"]: a for a in ann["annotations"]}
    def mask(row):
        a = by[int(row.ann_id)]
        h, w = img[a["image_id"]]
        seg = a["segmentation"]
        rle = MU.merge(MU.frPyObjects(seg, h, w)) if isinstance(seg, list) else (MU.frPyObjects(seg, h, w) if isinstance(seg["counts"], list) else seg)
        return MU.decode(rle)
    return mask


def check_E09(R):
    ex = R.extra("E09")
    d = m2_diff(R, ("E09", "e09_ads"), ("E09", "e09_fix"))
    rows = [("ads - fix (Overall)", f(d)), ("ads - fix (Part)", f(m2_diff(R, ("E09", "e09_ads"), ("E09", "e09_fix"), "part"))),
            ("ads - fix (Object)", f(m2_diff(R, ("E09", "e09_ads"), ("E09", "e09_fix"), "object")))]
    in_mask = {}
    if "e09_ads__points_b1" in ex:
        grid = ex["e09_fix__points_b1"]
        for blk in (0, 1):
            pa = ex[f"e09_ads__points_b{blk}"]
            disp = np.linalg.norm(pa - grid, axis=-1)
            outside = (~((pa >= 0) & (pa <= 1)).all(-1)).mean()
            cen = np.linalg.norm(pa - 0.5, axis=-1).mean()
            rows.append((f"block {blk}: mean |ads point - grid| (box units)", p(float(disp.mean()), 3)))
            rows.append((f"block {blk}: fraction of ads points outside the box", p(float(outside), 3)))
            rows.append((f"block {blk}: mean distance to box centre ads / fix", f"{cen:.3f} / {np.linalg.norm(grid - 0.5, axis=-1).mean():.3f}"))
        # fraction of points on the instance mask (1,500 val ROIs, rng 0)
        try:
            mk = _masks_for_val(R.dva)
            idx = np.random.default_rng(0).choice(len(R.dva), min(1500, len(R.dva)), replace=False)
            fr = {"ads": [], "fix": []}
            for i in idx:
                row = R.dva.iloc[i]
                m = mk(row)
                l, t, r, b = row.l, row.t, row.r, row.b
                for nm, pts in (("ads", ex["e09_ads__points_b1"][i]), ("fix", grid[i])):
                    x = np.clip((l + pts[:, 0] * (r - l)).astype(int), 0, m.shape[1] - 1)
                    y = np.clip((t + pts[:, 1] * (b - t)).astype(int), 0, m.shape[0] - 1)
                    fr[nm].append(m[y, x].mean())
            in_mask = {k: float(np.mean(v)) for k, v in fr.items()}
            rows.append(("fraction of last-block points on the instance mask ads / fix", f"{in_mask['ads']:.3f} / {in_mask['fix']:.3f}"))
        except Exception as e:  # noqa: BLE001
            rows.append(("instance-mask fraction", f"not computed ({e!r})"))
    ability = None if d is None else bool(d["lo"] > 0)
    return dict(ability=ability, rule="ads - fix val M2 Overall CI lo > 0 (point movement reported alongside)", rows=rows, in_mask=in_mask)


def check_E11(R):
    ex = R.extra("E11")
    g, c = ("E11", "e11_gate"), ("E11", "e11_concat")
    dpart = m2_diff(R, g, c, "part")
    rows = [("gate - concat (Overall)", f(m2_diff(R, g, c))), ("gate - concat (Part)", f(dpart)),
            ("gate - concat (Object)", f(m2_diff(R, g, c, "object"))),
            ("ctx-only - concat (Overall)", f(m2_diff(R, ("E11", "e11_ctx"), c)))]
    drop_part = None
    for rd in ("e11_gate", "e11_concat", "e11_ctx"):
        S0 = R.scores("E11", rd)
        if S0 is None:
            continue
        for cm in ("remove", "swap"):
            k = f"{rd}__ctx_{cm}_logits"
            if k not in ex:
                continue
            mc = R.m2(("E11", f"{rd}@{cm}"), ex[k])
            dd = {gg: summarize(R.m2(("E11", rd))[gg], mc[gg]) for gg in ("overall", "part")}
            sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
            known = R.Yva >= 0
            dp = float(np.abs(sig(S0) - sig(ex[k]))[known].mean())
            rows.append((f"{rd}: M2 drop when context {cm}d (Overall / Part)", f"{f(dd['overall'])} / {f(dd['part'])}"))
            rows.append((f"{rd}: mean |dp| when context {cm}d", p(dp, 4)))
            if rd == "e11_gate" and cm == "remove":
                drop_part = dd["part"]
    if "e11_gate__gate" in ex:
        gv = ex["e11_gate__gate"][:, 0]
        nin = np.load(RC.NIN_SG.format("val"))
        lv = R.dva.level.values
        for nm, m in (("object n_in 1-16", (lv == "object") & (nin <= 16)), ("object 17+", (lv == "object") & (nin > 16)),
                      ("part", lv == "part")):
            rows.append((f"mean local gate g [{nm}] (1 = local only)", p(float(gv[m].mean()), 3)))
    ability = None if dpart is None or drop_part is None else bool(dpart["lo"] > 0 and drop_part["lo"] > 0)
    return dict(ability=ability, rule="gate - concat Part M2 CI lo > 0 AND removing context lowers gate's Part M2 (CI lo > 0)", rows=rows)


def check_E12(R):
    t, s, b = ("ctrl_f672", "f672_ml_r2"), ("E12", "e12_cons"), BASE
    St, Ss, Sb = R.scores(*t), R.scores(*s), R.scores(*b)
    if St is None or Ss is None or Sb is None:
        return dict(ability=None, note="missing scores", rows=[])
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    pt = sig(St)

    def agree(S):     # per-ROI mean BCE(sigmoid(S) vs teacher prob) over 59 attrs, bootstrap over images
        q = np.clip(sig(S), 1e-6, 1 - 1e-6)
        per = -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean(1)
        W = R.boot.W
        return (W * per).sum(1) / W.sum(1)

    ag = summarize(agree(Ss), agree(Sb))
    teacher_gap = m2_diff(R, t, b)
    stud = m2_diff(R, s, b)
    closure = None
    if teacher_gap and stud and teacher_gap["point"] > 0:
        closure = stud["point"] / teacher_gap["point"]
    rows = [("teacher F672 - baseline F336 (val M2 Overall; does the teacher discriminate better?)", f(teacher_gap)),
            ("teacher F672 val M2 Overall / Part / Object", " / ".join(p(float(R.m2(t)[g][0])) for g in ("overall", "part", "object"))),
            ("student - baseline (val M2 Overall)", f(stud)),
            ("student - baseline (Part / Object)", f"{f(m2_diff(R, s, b, 'part'))} / {f(m2_diff(R, s, b, 'object'))}"),
            ("disagreement with teacher: BCE(student) - BCE(baseline) (lower = closer)", f(ag, 4)),
            ("gap closure = d(student) / d(teacher)", p(closure, 3))]
    ability = bool(ag["hi"] < 0 and closure is not None and closure > 0)
    return dict(ability=ability, rule="student closer to the teacher than the baseline (CI hi < 0) AND gap closure > 0", rows=rows)


CHECKS = {"E05": check_E05, "E08": check_E08, "E09": check_E09, "E11": check_E11, "E12": check_E12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--exclude", default="")
    a = ap.parse_args()
    data = RC.load_data(a.exclude or None)
    R = Run(a.run, data)
    L = ["# Shared-feature readout queue: report (PACO VAL only, seed 0, GPU 2+3 DDP)", "",
         f"Run: `{a.run}`. Primary = val M2 Overall at the last epoch vs the queue baseline **Multi-layer + R2 (blocks 4/8/12)** "
         "(`ctrl_f336/ml_r2`, defined for this queue, not a reproduction of the 11-04 run). Paired image bootstrap 95% CI, 1,000 reps. "
         "PACO test was not evaluated. Object 5-16 is not reported (official M2 on val is NaN).", ""]
    dc = json.load(open(os.path.join(a.run, "data_check", "report.json")))
    L += ["## Data check", "", f"- verified disjoint after exclusion: **{dc['ok']}**; train images excluded: {dc['n_train_excluded']} "
          f"(hits by key {dc['train_vs_eval_hits_by_key']}); left after exclusion {dc['train_vs_eval_after_exclusion']}",
          f"- queue reads: {', '.join(dc['used_by_queue'])}; F672 teacher = the same PACO train images (no other source)", ""]
    L += ["## Controls", "", "| run | status | val M2 Overall | Object | Part | Color | Material | Pattern | train-subset M2 |", "|---|---|---|---|---|---|---|---|---|"]
    out = {"controls": {}, "experiments": {}}
    for job, rd in (BASE, ("ctrl_f336", "r2_final"), ("ctrl_c0", "c0_r2"), ("ctrl_f672", "f672_ml_r2")):
        v = R.validity(job, rd)
        m = R.m2((job, rd))
        vals = [p(float(m[g][0])) if m else "n/a" for g in ("overall", "object", "part", "color", "material", "pattern_marking")]
        L.append(f"| {job}/{rd} | {v['status']}{'' if v['valid'] else ' (INVALID)'} | " + " | ".join(vals) + f" | {p(v['train_subset_M2'])} |")
        out["controls"][f"{job}/{rd}"] = dict(validity=v, M2={g: float(m[g][0]) for g in GROUPS} if m else None)
    d0 = os.path.join(a.run, "D0", "d0.json")
    if os.path.exists(d0):
        dd = json.load(open(d0))["decision"]
        L += ["", f"**D0**: slot 1 = **{dd['slot1']}** (binding problem confirmed = {dd['binding_problem_confirmed']}); "
              f"BD(ml_r2) {f(dd['BD_ml_r2'])} pp; BD gap vs C0 {f(dd['BD_gap_vs_c0'])}; score gap (PA_x) vs C0 {f(dd['score_gap_vs_c0'])}. Details: `D0/d0.md`."]
    slot1 = open(os.path.join(a.run, "slot1_used.txt")).read().strip() if os.path.exists(os.path.join(a.run, "slot1_used.txt")) else None
    L.append("")
    for e, spec in EXPS.items():
        if e in ("E05", "E08") and slot1 and e != slot1:
            continue
        st, _ = R.job_status(e)
        L += [f"## {e} — {spec['title']}", "", f"**Hypothesis.** {spec['hyp']}", "", f"**Training.** {spec['train']}", ""]
        L += ["| reader | status | epochs | train loss (ep1 → last) | val loss (ep0 → last) | val M2 by epoch | best-val epoch* | train-subset loss / M2 |", "|---|---|---|---|---|---|---|---|"]
        val = {}
        for rd in spec["readers"]:
            v = R.validity(e, rd)
            val[rd] = v
            tl = f"{v['train_loss'][0]} → {v['train_loss'][-1]}" if v["train_loss"] else "n/a"
            vl = f"{v['val_loss'][0]} → {v['val_loss'][-1]}" if v["val_loss"] else "n/a"
            L.append(f"| {rd} | {v['status']}{'' if v['valid'] else ' (INVALID)'} | {v['epochs_logged']} | {tl} | {vl} | {v['val_M2']} | {v['best_val_epoch']} | "
                     f"{p(v['train_subset_loss'], 4)} / {p(v['train_subset_M2'])} |")
        L += ["", "*best-val epoch is shown for information only; all numbers below use the last epoch (no selection on val).", ""]
        try:
            chk = CHECKS[e](R) if st == "done" else dict(ability=None, rows=[], note=f"job {st}")
        except Exception as ex:  # noqa: BLE001
            chk = dict(ability=None, rows=[], note=f"direct check failed: {ex!r}")
        L += [f"**Direct check** (rule: {chk.get('rule', '-')})", "", "| measure | value |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in chk.get("rows", [])]
        if chk.get("note"):
            L.append(f"| note | {chk['note']} |")
        L.append("")
        ds = downstream(R, (e, spec["main"])) if st == "done" else None
        L += [f"**Downstream** ({spec['main']} vs ml_r2, val M2, last epoch)", "", "| metric | baseline | method | Δ [95% CI] |", "|---|---|---|---|"]
        if ds:
            for g in GROUPS:
                L.append(f"| {g} | {ds['base'][g]:.2f} | {ds['abs'][g]:.2f} | {f(ds['delta'][g])} |")
        L.append("")
        valid = all(v["valid"] for v in val.values()) and R.validity(*BASE)["valid"]
        ver = dict(run_valid=valid, ability_shown=chk.get("ability"),
                   overall_improved=ds["overall_improved"] if ds else None,
                   concentrated_in_one_type=(f"{ds['concentrated']} (top {ds['top_type']}, share {p(ds['top_share'])})" if ds else None),
                   major_degradation=(ds["major_degradation"] or "none") if ds else None,
                   downstream_success=ds["downstream_success"] if ds else None)
        L += ["**Verdict**", "", f"- Training and evaluation completed normally: **{ver['run_valid']}**",
              f"- Expected ability shown in the direct check: **{ver['ability_shown']}**",
              f"- Val M2 Overall improved over Multi-layer + R2 (CI lo > 0): **{ver['overall_improved']}**",
              f"- Gain concentrated in one attribute type: **{ver['concentrated_in_one_type']}**",
              f"- Major degradation of another metric: **{ver['major_degradation']}**",
              f"- Downstream success (pre-registered): **{ver['downstream_success']}** — recorded separately from the hypothesis check.", ""]
        out["experiments"][e] = dict(spec=spec, validity=val, direct=chk, downstream=ds, verdict=ver)
    open(os.path.join(a.run, "REPORT.md"), "w").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(a.run, "report.json"), "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
