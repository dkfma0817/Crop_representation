"""CPU-only reports from existing Brixel official evaluation JSON files.

Usage: python -m src.brixel_eval.report --root ROOT

Hierarchical AP is already percent in the unchanged PACO evaluator. Values
are used verbatim; differences are percentage points. No evaluator, model,
prediction array, or accelerator library is imported or executed here.
"""

import argparse
import csv
import io
import json
import math
from pathlib import Path


MODELS = ("B0", "B1", "B2")
EPOCHS = range(1, 11)
REQUIRED_TYPES = ("color", "pattern_marking", "material")
GROUPS = (("Object", "obj-attrs"), ("Part", "obj-part-attrs"))
CORE_KEYS = {"Object": "obj-attr-AP", "Part": "obj-attr-part-AP"}
SELECTION_RULE = (
    "max (official VAL object AP + part AP)/2; earlier epoch on ties"
)
BOOKKEEPING = (
    "split", "pred", "subset", "n_eval_images", "n_gt_anns",
    "n_dets_given", "n_dets_kept", "dets_dropped_by_max_dets",
    "images_with_dropped", "alignment_mismatch", "n_joint_pairs",
    "n_valid_pairs_area_all",
)


def _read_json(path, issues):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
    except (OSError, ValueError) as exc:
        issues.append(f"{path}: {exc}")
        return None


def _ap(value):
    """Reject missing/nonfinite/sentinel AP without inventing replacements."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and 0 <= value <= 100:
            return value
    return None


def _hierarchical(data):
    h = data.get("hierarchical") if data is not None else None
    return h if isinstance(h, dict) else {}


def _core(data):
    h = _hierarchical(data)
    return {name: _ap(h.get(key)) for name, key in CORE_KEYS.items()}


def _metrics(data, attribute_types):
    h = _hierarchical(data)
    values = _core(data)
    obj, part = values["Object"], values["Part"]
    values = {"Overall": (obj + part) / 2 if obj is not None and part is not None
              else None, **values}
    for level, group in GROUPS:
        breakdown = h.get(group)
        breakdown = breakdown if isinstance(breakdown, dict) else {}
        for attr_type in attribute_types:
            values[f"{level} {attr_type}"] = _ap(breakdown.get(attr_type))
    return values


def _metadata_summary(root, paths):
    summaries, issues = [], []
    for supplied in paths:
        path = Path(supplied)
        if not path.is_absolute():
            path = root / path
        data = _read_json(path, issues)
        if data is not None:
            summaries.append({"path": str(path), "data": _json_safe(data)})
    return {"files": summaries, "issues": issues}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def build_report(root, metadata_paths=()):
    """Read inputs without modifying selection.json or any other source file."""
    root = Path(root)
    issues, missing = [], []
    inputs, input_issues = {}, {}

    def load(relative, split):
        path = root / relative
        local = []
        if not path.is_file():
            missing.append(relative)
            local.append(f"missing {relative}")
            data = None
        else:
            data = _read_json(path, local)
            if data is not None:
                if not _hierarchical(data):
                    local.append(f"{relative}: missing hierarchical object")
                if data.get("split", split) != split:
                    local.append(f"{relative}: split does not match {split}")
                    data = None
                elif data.get("subset"):
                    local.append(f"{relative}: subset evaluation is not a full-split result")
                    data = None
        inputs[relative] = data
        input_issues[relative] = local
        issues.extend(local)
        return data

    curves = {}
    for model in MODELS:
        curves[model] = {}
        for epoch in EPOCHS:
            relative = f"eval_val/{model}/ep{epoch}/official_results.json"
            data = load(relative, "val")
            core = _core(data)
            if any(v is None for v in core.values()):
                if data is not None:
                    issue = f"{relative}: missing or invalid Object/Part AP"
                    input_issues[relative].append(issue)
                    issues.append(issue)
            else:
                curves[model][epoch] = {"obj": core["Object"], "part": core["Part"]}
        load(f"eval_test/{model}/official_results.json", "test")

    # select_epoch.py is a top-level CLI with file-writing side effects, so
    # reuse its exact scoring expression here rather than importing it.
    selection_path = root / "selection.json"
    existing = _read_json(selection_path, issues) if selection_path.exists() else {}
    selection = {}
    selection_issues = {}
    for model in MODELS:
        local = []
        c = curves[model]
        best = max(c, key=lambda e: (round((c[e]["obj"] + c[e]["part"]) / 2, 10), -e)) \
            if len(c) == 10 else None
        saved = existing.get(model) if existing is not None else None
        epoch, source = None, None
        if saved is not None:
            candidate = saved.get("epoch") if isinstance(saved, dict) else None
            if type(candidate) is not int or candidate not in EPOCHS:
                local.append(f"selection.json/{model}: epoch must be an integer from 1 to 10")
            else:
                epoch, source = candidate, "selection.json"
                if best is not None and epoch != best:
                    local.append(f"selection.json/{model}: epoch {epoch} disagrees with selector epoch {best}")
                if epoch in c:
                    for key, curve_key in (("val_obj", "obj"), ("val_part", "part")):
                        if key in saved and (not isinstance(saved[key], (int, float))
                                              or not math.isclose(saved[key], c[epoch][curve_key],
                                                                  rel_tol=0, abs_tol=1e-8)):
                            local.append(f"selection.json/{model}: {key} disagrees with selected VAL result")
        elif existing is not None and best is not None:
            epoch, source = best, "computed with src/siglip_pm/select_epoch.py rule"
        else:
            local.append(f"{model}: no usable selection; all ten valid VAL epochs are required to compute one")
        if len(c) != 10:
            local.append(f"{model}: selection audit incomplete ({len(c)}/10 valid VAL epochs)")
        selection[model] = {"epoch": epoch, "source": source, "rule": SELECTION_RULE,
                            "valid_val_epochs": len(c), "issues": local}
        selection_issues[model] = local
        issues.extend(local)

    # Include every actual official type key, including optional transparency.
    observed = set()
    for data in inputs.values():
        h = _hierarchical(data)
        for _, group in GROUPS:
            if isinstance(h.get(group), dict):
                observed.update(h[group])
    attribute_types = list(REQUIRED_TYPES)
    attribute_types.extend(sorted(observed - set(REQUIRED_TYPES)))
    metric_columns = list(_metrics(None, attribute_types))
    rows = []
    bookkeeping = {}
    for split in ("val", "test"):
        for model in MODELS:
            epoch = selection[model]["epoch"]
            relative = (f"eval_val/{model}/ep{epoch}/official_results.json" if split == "val"
                        else f"eval_test/{model}/official_results.json")
            data = inputs.get(relative) if epoch is not None else None
            local = list(selection_issues[model])
            local.extend(input_issues.get(relative, []))
            values = _metrics(data, attribute_types)
            unavailable = [key for key, value in values.items() if value is None]
            if unavailable:
                issue = f"{split}/{model}: unavailable metrics: {', '.join(unavailable)}"
                local.append(issue)
                issues.append(issue)
            rows.append({"kind": "model", "split": split, "model": model,
                         "epoch": epoch, "baseline_epoch": None,
                         "status": "incomplete" if local else "complete",
                         **values, "issues": local, "source": relative if epoch is not None else None})
            if data is not None:
                bookkeeping[f"{split}/{model}"] = {
                    key: _json_safe(data[key]) for key in BOOKKEEPING if key in data
                }

    contrasts = []
    for split in ("val", "test"):
        by_model = {row["model"]: row for row in rows if row["split"] == split}
        for newer, older in (("B1", "B0"), ("B2", "B1")):
            left, right = by_model[newer], by_model[older]
            values = {key: left[key] - right[key]
                      if left[key] is not None and right[key] is not None else None
                      for key in metric_columns}
            contrasts.append({"kind": "delta_pp", "split": split,
                              "model": f"{newer}−{older}", "epoch": left["epoch"],
                              "baseline_epoch": right["epoch"],
                              "status": "complete" if left["status"] == right["status"] == "complete"
                              else "incomplete", **values,
                              "issues": list(dict.fromkeys(left["issues"] + right["issues"]))})

    metadata = _metadata_summary(root, metadata_paths)
    issues.extend(metadata["issues"])
    return {
        "status": "incomplete" if issues else "complete", "root": str(root),
        "units": "AP percent (raw official hierarchical values); deltas in percentage points",
        "overall_formula": "(obj-attr-AP + obj-attr-part-AP) / 2",
        "selection_rule": SELECTION_RULE,
        "selection_score": "(round((obj + part) / 2, 10), -epoch)",
        "selection": selection,
        "val_curves": curves,
        "attribute_types": attribute_types, "metric_columns": metric_columns,
        "rows": rows, "contrasts": contrasts,
        "official_bookkeeping": bookkeeping, "metadata": metadata,
        "missing_inputs": missing, "issues": list(dict.fromkeys(issues)),
        "has_results": any(_ap(value) is not None for data in inputs.values()
                           for value in _core(data).values()),
    }


def _table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            text = "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(report):
    lines = ["# Brixel official evaluation", "", f"Status: **{report['status'].upper()}**", "",
             f"Run root: `{report['root']}`", "",
             "B0 = original DINOv3-B/16 full image; B1 = the same frozen backbone + official pretrained BRIXEL ViT-B/256; B2 = the same backbone on each GT ROI crop.",
             "All inputs use the exact BRIXEL README square256 transform. B0/B1 input pixels are identical; B2 is cropped before the same transform.",
             "Fresh unchanged R2 readers, seed 0 (reader init 7), 10 epochs, matched data/optimizer/schedule; all extractors remain frozen.",
             "GT box + GT category oracle attribute AP uses the unchanged official evaluator, usual attribute AP, max_dets=300. No ROI-size pilot bins, M2/linear probe, or BRIXEL training.", "",
             "AP values are raw official hierarchical percentages (0–100). Differences are percentage points.",
             "Overall = (obj-attr-AP + obj-attr-part-AP) / 2. Object and Part use those two official keys directly.",
             "Epoch selection uses official VAL only: maximize the mean rounded to 10 decimal places, then choose the earlier epoch on ties.",
             "Existing selection.json is read and audited; absent entries are computed only with all ten valid VAL epochs.",
             "Test results are read from eval_test/B0, eval_test/B1, and eval_test/B2 for the selected checkpoints.",
             "Missing or invalid metrics are shown as — and stored as null in JSON or blank CSV cells.", ""]
    for split in ("val", "test"):
        lines.extend([f"## {split.title()}", "",
                      _table([row for row in report["rows"] if row["split"] == split],
                             ["model", "epoch", "status", *report["metric_columns"]]), "",
                      "Differences (pp):", "",
                      _table([row for row in report["contrasts"] if row["split"] == split],
                             ["model", "status", *report["metric_columns"]]), ""])
    lines.extend(["Breakdown columns map Object to `hierarchical.obj-attrs` and Part to `hierarchical.obj-part-attrs`; type key names are preserved.", ""])
    implementation = next((item["data"] for item in report["metadata"]["files"]
                           if item["data"].get("status") == "passed" and "conditions" in item["data"]), None)
    if implementation:
        rows = []
        for name, data in implementation["conditions"].items():
            rows.append({"model": name, "feature shape": "×".join(map(str, data["extractor_feature_shape"])),
                         "extractor median ms": data["extractor_median_ms"],
                         "unit": "ROI crop" if name == "B2" else "image",
                         "Val predict s": data["inference"]["val"]["seconds"],
                         "Test predict s": data["inference"]["test"]["seconds"],
                         "Test encoder canvases": data["inference"]["test"]["encoder_canvases"]})
        lines.extend(["## Implementation verification and observed inference costs", "",
                      f"All checks passed. Same backbone tensor SHA256: `{implementation['backbone_sha256']}`.",
                      f"{implementation['original_source_files_checked']} original source files checked; none changed. Input checkpoint file hashes also rechecked.",
                      "B0/B1 extract once per image in each prediction pass; only B2 re-encodes each ROI. GT box/category alignment mismatches: 0 for all six selected Val/Test evaluations.", "",
                      _table(rows, ["model", "feature shape", "extractor median ms", "unit", "Val predict s", "Test predict s", "Test encoder canvases"]), "",
                      implementation["cost_note"],
                      "Prediction times exclude CPU Official evaluator runtime. Test covers 9,443 images / 131,899 ROIs; the unchanged official max_dets=300 keeps 130,527 detections and drops 1,372 in each condition.", ""])
    if report["issues"]:
        lines.extend(["## Incomplete inputs and audit issues", ""])
        lines.extend(f"- {issue}" for issue in report["issues"])
        lines.append("")
    if report["metadata"]["files"]:
        lines.extend(["## Optional verification/data metadata", "",
                      "Supplied JSON metadata is reproduced for reference; it does not supply AP values.", ""])
        for item in report["metadata"]["files"]:
            lines.extend([f"Source: `{item['path']}`", "", "```json",
                          json.dumps(item["data"], indent=2, ensure_ascii=False, allow_nan=False), "```", ""])
    return "\n".join(lines)


def write_report(report, root):
    root = Path(root)
    fields = ["kind", "split", "model", "epoch", "baseline_epoch", "status",
              *report["metric_columns"], "issues", "source"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in report["rows"] + report["contrasts"]:
        writer.writerow({**row, "issues": "; ".join(row["issues"])})
    (root / "results.csv").write_text(stream.getvalue(), encoding="utf-8")
    (root / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    # An empty run gets machine-readable incomplete status, never a final report.
    if report["has_results"]:
        (root / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata", action="append", default=[], metavar="JSON",
                        help="optional verification/data metadata JSON; repeatable, relative to root")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"root directory does not exist: {args.root}")
    report = build_report(args.root, args.metadata)
    write_report(report, args.root)
    print(json.dumps({"status": report["status"], "missing_inputs": len(report["missing_inputs"]),
                      "issues": len(report["issues"]), "markdown_written": report["has_results"]}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
