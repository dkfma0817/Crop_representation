"""All filesystem locations in one place. Every path can be overridden with the environment variable of the same name.

Layout (repo root):
  cache/          frozen-model files, ROI lists, feature cache, runs, probes   (gitignored; was raev2_crop_recon/perception_cache)
  third_party/    RAEv2 checkout                                              (gitignored)
  experiments/raev2_crop_recon/{results,figures}   committed outputs
  logs/           run logs and flags                                          (gitignored)
"""
import os
from pathlib import Path


def _p(name, default):
    return str(Path(os.environ.get(name, default)))


ROOT = str(Path(__file__).resolve().parents[1])

# --- project folders -------------------------------------------------------------------------------------------------
CACHE = _p("CACHE", os.path.join(ROOT, "cache"))
EXP = _p("EXP", os.path.join(ROOT, "experiments", "raev2_crop_recon"))
RES = os.path.join(EXP, "results")
FIG = os.path.join(EXP, "figures")
LOGS = _p("LOGS", os.path.join(ROOT, "logs"))
RAEV2 = _p("RAEV2", os.path.join(ROOT, "third_party", "RAEv2"))

# --- datasets on this server -----------------------------------------------------------------------------------------
COCO_TRAIN = _p("COCO_TRAIN", "/home/irteam/datasets/coco/train2017")
COCO_ANN = _p("COCO_ANN", "/home/irteam/datasets/coco/annotations/instances_train2017.json")
PACO_DIR = _p("PACO_DIR", "/home/irteam/datasets/paco")
IMAGENET_VAL = _p("IMAGENET_VAL", "/dataset/_imagenet_val_extracted_5000")          # not on this server (verify_pair only)

# --- derived files (cache/) ------------------------------------------------------------------------------------------
COCO_ROIS = os.path.join(CACHE, "data", "coco_rois.parquet")          # src/data/make_rois.py
RAE_ROIS = os.path.join(CACHE, "data", "rae_rois.parquet")            # src/data/make_rois.py
PACO_VALTEST_IDS = os.path.join(CACHE, "data", "paco_valtest_ids.json")
STREAM = os.path.join(CACHE, "stream", "stream_ids_12000.npy")        # src/data/make_stream.py
FEAT = os.path.join(CACHE, "feat")                                    # src/data/extract_cache.py
INIT = os.path.join(CACHE, "init", "coordmlp_init_s0.pt")             # python -m src.engine.train make_init
RUNS = os.path.join(CACHE, "runs")
PROBE = os.path.join(CACHE, "probe")
MODELS = os.path.join(CACHE, "raev2_models", "stage1", "imagenet", "dinov2b-k1")
TORCH_HOME = os.path.join(CACHE, "torch_hub")

# --- code / data from earlier projects (old server, NOT on this server) ----------------------------------------------
_OLD = "/workspace/Areum/research"
PACO_LP = _p("PACO_LP", f"{_OLD}/2026-10-24/dinov2_oracle_region_decoder/perception_cache/paco_lp")   # PACO manifests/labels
RAE_PILOT = _p("RAE_PILOT", f"{_OLD}/2026-10-26/rae_crop_pilot")                                     # _pylib
PACO_READER_CODE = _p("PACO_READER_CODE", f"{_OLD}/2026-10-30/paco_region_reader/code")              # common_rr
FEAT_UPSAMPLE_CODE = _p("FEAT_UPSAMPLE_CODE", f"{_OLD}/2026-10-31/paco_feat_upsample/code")          # eval_fu
