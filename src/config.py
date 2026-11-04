"""Filesystem locations, read from configs/paths.yaml (or the YAML named by $CROP_PATHS).

Relative paths in the YAML are resolved against the repo root; every key can be overridden by the environment variable of the
same name in upper case (e.g. COCO_TRAIN=/data/coco/train2017). The module exposes the paths as constants so the rest of the
code just does `from src.config import CACHE, COCO_ROIS, ...`.
"""
import os
from pathlib import Path

import yaml

ROOT = str(Path(__file__).resolve().parents[1])
PATHS_YAML = os.environ.get("CROP_PATHS", os.path.join(ROOT, "configs", "paths.yaml"))
with open(PATHS_YAML) as _f:
    _Y = yaml.safe_load(_f)


def _resolve(key, value, base=ROOT):
    value = os.environ.get(key.upper(), value)
    return str(Path(value) if os.path.isabs(value) else Path(base, value))


# --- project folders -------------------------------------------------------------------------------------------------
CACHE = _resolve("cache", _Y["project"]["cache"])
EXP = _resolve("exp", _Y["project"]["exp"])
LOGS = _resolve("logs", _Y["project"]["logs"])
OUTPUT = _resolve("output", _Y["project"]["output"])
RAEV2 = _resolve("raev2", _Y["project"]["raev2"])
RES = os.path.join(EXP, "results")
FIG = os.path.join(EXP, "figures")

# --- datasets --------------------------------------------------------------------------------------------------------
COCO_TRAIN = _resolve("coco_train", _Y["datasets"]["coco_train"])
COCO_VAL = _resolve("coco_val", _Y["datasets"]["coco_val"])
COCO_ANN = _resolve("coco_ann", _Y["datasets"]["coco_ann"])
PACO_DIR = _resolve("paco_dir", _Y["datasets"]["paco_dir"])
IMAGENET_VAL = _resolve("imagenet_val", _Y["datasets"]["imagenet_val"])

# --- files under cache/ ----------------------------------------------------------------------------------------------
_C = {k: _resolve(k, v, CACHE) for k, v in _Y["cache_files"].items()}
COCO_ROIS = _C["coco_rois"]
RAE_ROIS = _C["rae_rois"]
PACO_VALTEST_IDS = _C["paco_valtest_ids"]
STREAM = _C["stream"]
FEAT = _C["feat"]
INIT = _C["init"]
RUNS = _C["runs"]
PROBE = _C["probe"]
MODELS = _C["models"]
TORCH_HOME = _C["torch_hub"]

# --- data / code copied from earlier projects ------------------------------------------------------------------------
RESEARCH_ROOT = _resolve("research_root", _Y["legacy"]["research_root"])
os.environ.setdefault("RESEARCH_ROOT", RESEARCH_ROOT)   # read by the copied old-project code instead of /workspace/Areum/research
PACO_LP = _resolve("paco_lp", _Y["legacy"]["paco_lp"])
RAE_PILOT = _resolve("rae_pilot", _Y["legacy"]["rae_pilot"])
PACO_READER_CODE = _resolve("paco_reader_code", _Y["legacy"]["paco_reader_code"])
FEAT_UPSAMPLE_CODE = _resolve("feat_upsample_code", _Y["legacy"]["feat_upsample_code"])
