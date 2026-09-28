"""Parse the `train_opt.txt` options dump that main.py writes into every run's log dir.

Checkpoints do not store hyperparameters (the model is rebuilt from CLI options), so the
evaluation scripts rebuild the training-time options from this file instead.
"""
import ast
import re


def _coerce(v):
    v = v.strip()
    if v in ("None", ""):   return None
    if v == "True":         return True
    if v == "False":        return False
    if v[:1] in "[{(":
        try:    return ast.literal_eval(v)
        except Exception: return v
    try:    return int(v)
    except ValueError: pass
    try:    return float(v)
    except ValueError: pass
    return v


def opts_from_train_opt(path):
    """Parse a version_N/train_opt.txt options dump into a kwargs dict."""
    opts = {}
    for line in open(path):
        line = line.rstrip("\n")
        if ":" not in line or set(line.strip()) <= set("- "):
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        if not key or " " in key:            # header / separator lines
            continue
        val = re.split(r"\s+\[default", rest)[0]     # drop the "[default: ...]" annotation
        opts[key] = _coerce(val)
    return opts
