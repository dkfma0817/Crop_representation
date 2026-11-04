"""Small YAML config loader.

  load_yaml(path)  -> dict; a top-level `_base_: other.yaml` (relative to the file) is loaded first and overridden key by key.
  with_config(ap)  -> parse argv with `--config FILE`: YAML values become argparse defaults, CLI flags still win, and
                      `required=True` arguments may come from the YAML instead of the command line.
"""
import argparse
import os
import sys

import yaml


def load_yaml(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        merged = load_yaml(os.path.join(os.path.dirname(os.path.abspath(path)), base))
        merged.update(cfg)
        cfg = merged
    return cfg


def with_config(ap: argparse.ArgumentParser, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    ap.add_argument("--config", default=None, help="YAML file; its keys are argument names (CLI overrides)")
    if known.config:
        cfg = load_yaml(known.config)
        dests = {a.dest: a for a in ap._actions}
        unknown = sorted(set(cfg) - set(dests))
        if unknown:
            ap.error(f"{known.config}: unknown keys {unknown}")
        for k in cfg:
            dests[k].required = False
        ap.set_defaults(**cfg)
    return ap.parse_args(argv)
