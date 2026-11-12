"""Consolidate implementation evidence; never evaluates model quality or changes selection."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    project = Path(__file__).resolve().parents[2]
    read = lambda p: json.loads((root / p).read_text())
    selected = read("selection.json")
    conversion = read("verify/backbone_conversion.json")
    compact = read("verify/reader_compaction.json")
    matched = read("verify/matched_conditions.json")
    assert conversion["all_passed"] and compact["status"] == "passed" and matched["reader_settings_identical"]
    source = (project / "src/brixel_eval/train.py").read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "roi_chunks")
    assert hashlib.sha256((ast.get_source_segment(source, function) + "\n").encode()).hexdigest() == compact["roi_chunks_sha256"]
    records = {}
    for name in ("B0", "B1", "B2"):
        audit = read(f"verify/checkpoints_{name}.json")
        frozen = read(f"verify/frozen_{name}.json")
        assert audit["backbone_sha256"] == matched["backbone_sha256"]
        assert audit["all_frozen"] and audit["eval_mode"]
        assert hashlib.sha256(Path(audit["timm_weights"]).read_bytes()).hexdigest() == audit["timm_file_sha256"]
        if name == "B1":
            assert hashlib.sha256(Path(audit["adapter_weights"]).read_bytes()).hexdigest() == audit["adapter_file_sha256"]
        assert frozen["unchanged"] and frozen["before"] == frozen["after"]
        epoch = selected[name]["epoch"]
        infer, official = {}, {}
        for split, images, rows in (("val", 2410, 31919), ("test", 9443, 131899)):
            item = read(f"verify/infer_{name}_{split}_ep{epoch}.json")
            assert item["rows"] == rows and item["images"] == images
            assert item["encoder_canvases"] == (rows if name == "B2" else images)
            assert item["feature_token_shapes"] == [[4096 if name == "B1" else 256, 768]]
            infer[split] = {k: item[k] for k in ("rows", "images", "encoder_canvases", "seconds")}
            path = f"eval_val/{name}/ep{epoch}/official_results.json" if split == "val" else f"eval_test/{name}/official_results.json"
            result = read(path)
            assert result["alignment_mismatch"] == 0
            assert not result["subset"]
            assert result["n_eval_images"] == images and result["n_dets_given"] == rows
            official[split] = {k: result[k] for k in ("alignment_mismatch", "n_dets_given", "n_dets_kept", "dets_dropped_by_max_dets", "seconds")}
        cost = read(f"verify/cost_{name}.json")
        records[name] = dict(selected_epoch=epoch, extractor_feature_shape=cost["feature_shape"],
                             extractor_median_ms=cost["median_latency_ms"],
                             extractor_forward_peak_bytes_with_reader_resident=cost["peak_allocated_bytes"],
                             extractor_unit=cost["unit"], all_frozen=True, unchanged_after_training=True,
                             inference=infer, official=official)
    originals = read("verify/original_files.json")["source_sha256"]
    changed = [name for name, value in originals.items()
               if hashlib.sha256((project / name).read_bytes()).hexdigest() != value]
    assert not changed, changed
    (root / "verify/original_files_unchanged.json").write_text(json.dumps(dict(n_files=len(originals), changed=changed), indent=2) + "\n")
    summary = dict(status="passed", backbone_sha256=matched["backbone_sha256"],
                   original_source_files_checked=len(originals), original_source_files_changed=changed,
                   checkpoint_load="strict; adapter/head 213 tensors, same converted pretrained backbone in B0/B1/B2",
                   backbone_provenance="cached timm DINOv3-B inference weights converted to official keys; discarded unused/deterministic tensors reconstructed; CPU features bitwise identical to timm; original cache unchanged",
                   preprocessing="exact BRIXEL README square256 v2 transform; B0/B1 pixel tensors equal; B2 tight crop then same transform",
                   geometry="native grid16 stride16 (B0/B2), grid64 stride4 (B1); anisotropic ROI mapping verified analytically",
                   reader="unchanged R2, fresh per condition, identical initial hash/settings/data/optimizer; seed0 and init seed7",
                   compaction="masked-out keys only; CPU/H200 logits and gradients verified; FP32/BF16 numerical differences recorded in reader_compaction.json",
                   selection="unchanged official-Val selector, max (Object+Part)/2, earlier on ties; fixed Test epochs",
                   cost_note="extractor batch1 CUDA event measurements on shared H200 GPUs; reader parameters remain resident, but only extractor forward is timed; B2 unit is one ROI. These are observed costs, not exclusive-device comparative benchmarks. Whole-split prediction wall times include I/O and R2; see raw latency samples",
                   conversion_checks_passed=conversion["all_passed"], compaction_evidence="verify/reader_compaction.json",
                   conditions=records)
    path = root / "verify/implementation.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status="passed", output=str(path))))


if __name__ == "__main__":
    main()
