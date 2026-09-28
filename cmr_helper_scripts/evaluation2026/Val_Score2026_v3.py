#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gc
import json
import time
import uuid
import shutil
import gzip
import tarfile
import zipfile
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

try:
    import rarfile  # type: ignore
    _HAS_RAR = True
except Exception:
    rarfile = None
    _HAS_RAR = False

# Project utilities
from cmr_helper_scripts.evaluation2026.utils_bgc import *
from cmr_helper_scripts.evaluation2026.utils_datasl import *
from cmr_helper_scripts.evaluation2026.utils_flow import *
from cmr_helper_scripts.evaluation2026.utils_metrics import *


# ============================================================
# Task configuration
# ============================================================

TASK_CONFIG = {
    "R1": {
        "dir_name": "TaskR1R2",
        "detail_metrics": ["SSIM", "nRMSE", "RelErr", "AngErr"],
        "summary_metrics": ["SSIM", "nRMSE", "RelErr", "AngErr"],
        "summary_level": "Center",
        "needs_flowvn": False,
    },
    "R2": {
        "dir_name": "TaskR1R2",
        "detail_metrics": [
            "ComplexDiffErr",
            # "ComplexDiffErr_FlowVN",
            "ComplexDiffErr_Diff",
            "BetterThanFlowVN",
        ],
        "summary_metrics": ["ComplexDiffErr", "ComplexDiffErr_Diff"],
        "summary_level": "Center",
        "needs_flowvn": True,
    },
    "S1": {
        "dir_name": "TaskS1",
        "detail_metrics": ["SSIM", "nRMSE", "RelErr", "AngErr"],
        "summary_metrics": ["SSIM", "nRMSE", "RelErr", "AngErr"],
        "summary_level": "Center",
        "needs_flowvn": False,
    },
    "S2": {
        "dir_name": "TaskS2",
        "detail_metrics": ["SSIM", "nRMSE", "RelErr", "AngErr"],
        "summary_metrics": ["SSIM", "nRMSE", "RelErr", "AngErr"],
        "summary_level": "Anatomy",
        "needs_flowvn": False,
    },
}

MISSING_FILLS = {
    "SSIM": 0.0,
    "nRMSE": 1.0,
    "RelErr": 1.0,
    "AngErr": 1.0,
    "ComplexDiffErr": 1.0,
    "ComplexDiffErr_Diff": 1.0,
}

NPZ_NAME_RE = re.compile(r"^img_ktGaussian(?P<R>[^/\\]+)\.npz$")


# ============================================================
# Optional global SSIM cache
# ============================================================

_GLOBAL_SSIM3D = None


def get_cached_ssim3d():
    """
    Cache a single SSIM3D instance if utils_metrics exposes SSIM3D class.
    If not available, return None and fall back to SSIM(...) function.
    """
    global _GLOBAL_SSIM3D
    if _GLOBAL_SSIM3D is not None:
        return _GLOBAL_SSIM3D

    try:
        if "SSIM3D" in globals():
            _GLOBAL_SSIM3D = SSIM3D()
            return _GLOBAL_SSIM3D
    except Exception:
        pass
    return None


# ============================================================
# torch inference context
# ============================================================

class _NullContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def inference_context():
    if _HAS_TORCH:
        try:
            return torch.inference_mode()
        except Exception:
            return torch.no_grad()
    return _NullContext()


# ============================================================
# SampleID
# ============================================================

@dataclass(frozen=True)
class SampleID:
    dir_name: str
    settype: str
    anatomy: str
    center: str
    vendor: str
    patient: str
    r: str

    @property
    def rel_pred_path(self) -> str:
        return (
            f"{self.dir_name}/{self.settype}/{self.anatomy}/"
            f"{self.center}/{self.vendor}/{self.patient}/"
            f"img_ktGaussian{self.r}.npz"
        )


# ============================================================
# Path helpers
# ============================================================

def norm_relpath(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def list_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            out.append(os.path.join(dirpath, fn))
    return out


def extract_sample_id_from_relpath(relpath: str, dir_name: str) -> Optional[SampleID]:
    """
    Expect:
      {dir_name}/{ValidationSet}/{Anatomy}/{Center}/{Vendor}/PXXX/img_ktGaussian{R}.npz
    """
    parts = relpath.split("/")
    if len(parts) != 7:
        return None
    t, settype, anatomy, center, vendor, patient, fname = parts
    if t != dir_name:
        return None
    m = NPZ_NAME_RE.match(fname)
    if m is None:
        return None
    return SampleID(
        dir_name=t,
        settype=settype,
        anatomy=anatomy,
        center=center,
        vendor=vendor,
        patient=patient,
        r=m.group("R"),
    )


# ============================================================
# Archive extraction to temp
# ============================================================

def _is_archive(path: str) -> bool:
    low = path.lower()
    return low.endswith((".zip", ".tar", ".tgz", ".tar.gz", ".gz", ".rar"))


def extract_to_temp(input_path: str, output_dir: str) -> Tuple[str, Optional[str]]:
    """
    If input_path is a directory → (input_path, None).
    If archive → extract into {output_dir}/temp/<uuid>/ → (extracted_root, temp_dir).
    """
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)

    if os.path.isdir(input_path):
        return input_path, None
    if not os.path.isfile(input_path):
        raise RuntimeError(f"--input not found: {input_path}")
    if not _is_archive(input_path):
        raise RuntimeError(
            f"--input is not a directory and not a supported archive: {input_path}"
        )

    temp_base = os.path.join(output_dir, "temp")
    os.makedirs(temp_base, exist_ok=True)
    temp_dir = os.path.join(temp_base, f"extract_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)

    low = input_path.lower()

    def _safe_tar(tar: tarfile.TarFile, path: str) -> None:
        abs_root = os.path.abspath(path)
        for member in tar.getmembers():
            abs_target = os.path.abspath(os.path.join(path, member.name))
            if not abs_target.startswith(abs_root + os.sep) and abs_target != abs_root:
                raise RuntimeError(f"Unsafe tar path traversal: {member.name}")
        tar.extractall(path=path)

    try:
        if low.endswith(".zip"):
            with zipfile.ZipFile(input_path, "r") as z:
                z.extractall(temp_dir)

        elif low.endswith((".tar", ".tgz", ".tar.gz")):
            with tarfile.open(input_path, "r:*") as tar:
                _safe_tar(tar, temp_dir)

        elif low.endswith(".gz") and not low.endswith(".tar.gz"):
            out_file = os.path.join(temp_dir, os.path.basename(input_path[:-3]))
            with gzip.open(input_path, "rb") as fi, open(out_file, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            if out_file.lower().endswith(".tar"):
                with tarfile.open(out_file, "r:*") as tar:
                    _safe_tar(tar, temp_dir)
                os.remove(out_file)

        elif low.endswith(".rar"):
            if not _HAS_RAR:
                raise RuntimeError(
                    "rarfile is not available. "
                    "Please `pip install rarfile` and ensure unrar is installed."
                )
            with rarfile.RarFile(input_path) as r:
                r.extractall(temp_dir)

        else:
            raise RuntimeError(f"Unsupported archive type: {input_path}")

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return temp_dir, temp_dir


# ============================================================
# Loaders / dtype helpers
# ============================================================

def as_float32_if_real(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if np.iscomplexobj(x):
        if x.dtype == np.complex64:
            return x
        return x.astype(np.complex64, copy=False)
    else:
        if x.dtype == np.float32:
            return x
        return x.astype(np.float32, copy=False)


def load_segmask_mat(path: str) -> np.ndarray:
    seg = load_mat(path, key="segmask")[:]
    return np.asarray(seg, dtype=np.float32)


def load_npz_dense(path: str) -> np.ndarray:
    arr = load_coo_npz(path, as_dense=True)
    return as_float32_if_real(arr)


# ============================================================
# ROI bounding box helpers
# ============================================================

def get_nonzero_bbox(mask: np.ndarray) -> Optional[Tuple[slice, ...]]:
    """
    Return a minimal bounding-box slice tuple that covers nonzero entries.
    Works for 2D/3D/4D+ masks.
    """
    if mask is None:
        return None

    mask_nz = np.asarray(mask) != 0
    if not np.any(mask_nz):
        return None

    coords = np.nonzero(mask_nz)
    bbox = tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)
    return bbox


def crop_arrays_to_seg_bbox(
    gt: np.ndarray,
    pred: np.ndarray,
    segmask: np.ndarray,
    pred_fvn: Optional[np.ndarray] = None,
    corr_maps: Optional[np.ndarray] = None,
):
    """
    Crop arrays according to segmask non-zero bbox.
    Assumes segmask spatial dims correspond to the trailing dims of arrays.
    """
    bbox = get_nonzero_bbox(segmask)
    if bbox is None:
        if pred_fvn is None:
            return gt, pred, segmask, corr_maps
        return gt, pred, segmask, pred_fvn, corr_maps

    seg_ndim = segmask.ndim
    gt_ndim = gt.ndim

    if seg_ndim > gt_ndim:
        raise RuntimeError(
            f"segmask ndim ({seg_ndim}) > data ndim ({gt_ndim}), cannot crop"
        )

    prefix = (slice(None),) * (gt_ndim - seg_ndim)
    full_slice = prefix + bbox

    gt_c = gt[full_slice]
    pred_c = pred[full_slice]
    seg_c = segmask[bbox]

    if pred_fvn is None:
        corr_c = corr_maps[full_slice] if corr_maps is not None else None
        return gt_c, pred_c, seg_c, corr_c

    fvn_c = pred_fvn[full_slice]
    corr_c = corr_maps[full_slice] if corr_maps is not None else None
    return gt_c, pred_c, seg_c, fvn_c, corr_c


# ============================================================
# Metric computation
# ============================================================

def get_or_create_corrmap(
    gt: np.ndarray,
    segmask: np.ndarray,
    gt_base: str,
) -> np.ndarray:
    """
    Load cached corrmap from the case directory if it exists;
    otherwise compute it from GT, multiply by segmask, and save as corrmap.npz.
    """
    corrmap_path = os.path.join(gt_base, "corrmap.npz")

    if os.path.exists(corrmap_path):
        corr_maps = load_coo_npz(corrmap_path, as_dense=True)
        corr_maps = as_float32_if_real(corr_maps)
    else:
        corr_maps = execute_MSAC(gt, corr_fit_order=3, th=0.1)
        corr_maps = np.asarray(corr_maps, dtype=np.float32) * np.asarray(segmask, dtype=np.float32)
        save_coo_npz(corrmap_path, corr_maps)

    return corr_maps


def _phase_correct_inplace_copy(
    gt: np.ndarray,
    *preds: np.ndarray,
    corr_maps: Optional[np.ndarray] = None
):
    """
    Apply MSAC background-phase correction to GT and any number of predictions.

    Notes:
    - Keeps original inputs unchanged.
    - Uses one exp factor.
    - Avoids deeper unnecessary copies.
    """
    if corr_maps is None:
        corr_maps = execute_MSAC(gt, corr_fit_order=3, th=0.1)
        corr_maps = np.asarray(corr_maps, dtype=np.float32)

    factor = np.exp(-1j * corr_maps).astype(np.complex64, copy=False)

    gt_c = np.array(gt, copy=True)
    gt_c = as_float32_if_real(gt_c)
    gt_c[1:] *= factor

    results = [gt_c]

    for p in preds:
        pc = np.array(p, copy=True)
        pc = as_float32_if_real(pc)
        pc[1:] *= factor
        results.append(pc)

    return results


def compute_ssim_cached(mag_pred: np.ndarray, mag_gt: np.ndarray, segmask: np.ndarray) -> float:
    """
    Prefer cached SSIM3D object if available in utils_metrics, otherwise fallback to SSIM().
    """
    ssim_obj = get_cached_ssim3d()
    if ssim_obj is not None:
        try:
            return float(ssim_obj(mag_pred, mag_gt, segmask))
        except Exception:
            pass
    return float(SSIM(mag_pred, mag_gt, segmask))


def compute_metrics_standard(
    gt: np.ndarray,
    pred: np.ndarray,
    segmask: np.ndarray,
    corr_maps: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """For R1 / S1 / S2: SSIM, nRMSE, RelErr, AngErr."""
    with inference_context():
        gt, pred, segmask, corr_maps = crop_arrays_to_seg_bbox(
            gt, pred, segmask, pred_fvn=None, corr_maps=corr_maps
        )

        gt_c, pred_c = _phase_correct_inplace_copy(gt, pred, corr_maps=corr_maps)

        mag_gt, flow_gt = complex2magflow(gt_c)
        mag_pred, flow_pred = complex2magflow(pred_c)

        mag_gt = as_float32_if_real(mag_gt)
        mag_pred = as_float32_if_real(mag_pred)
        flow_gt = as_float32_if_real(flow_gt)
        flow_pred = as_float32_if_real(flow_pred)
        segmask = np.asarray(segmask, dtype=np.float32)

        return {
            "SSIM": compute_ssim_cached(mag_pred, mag_gt, segmask),
            "nRMSE": float(nRMSE(mag_pred, mag_gt, segmask)),
            "RelErr": float(RelErr(flow_pred, flow_gt, segmask)),
            "AngErr": float(AngErr(flow_pred, flow_gt, segmask)),
        }


def compute_metrics_r2(
    gt: np.ndarray,
    pred_sub: np.ndarray,
    pred_fvn: np.ndarray,
    segmask: np.ndarray,
    corr_maps: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """For R2: ComplexDiffErr for submission AND FlowVN baseline."""
    with inference_context():
        gt, pred_sub, segmask, pred_fvn, corr_maps = crop_arrays_to_seg_bbox(
            gt, pred_sub, segmask, pred_fvn=pred_fvn, corr_maps=corr_maps
        )

        gt_c, sub_c, fvn_c = _phase_correct_inplace_copy(
            gt, pred_sub, pred_fvn, corr_maps=corr_maps
        )
        segmask = np.asarray(segmask, dtype=np.float32)

        cde_sub = float(ComplexDiffErr(sub_c, gt_c, segmask))
        cde_fvn = float(ComplexDiffErr(fvn_c, gt_c, segmask))
        return {
            "ComplexDiffErr": cde_sub,
            "ComplexDiffErr_FlowVN": cde_fvn,
            "ComplexDiffErr_Diff": cde_sub - cde_fvn,
            "BetterThanFlowVN": "Yes" if cde_sub < cde_fvn else "No",
        }


# ============================================================
# Adjusted mean (penalize missing submissions)
# ============================================================

def adjusted_mean(
    values: List[float], expected_total: int, missing_fill: float
) -> float:
    if expected_total <= 0:
        return float("nan")
    missing = expected_total - len(values)
    return (float(np.sum(values, dtype=np.float64)) + missing * float(missing_fill)) / float(expected_total)


def fill_metrics_with_missing_values(rec: Dict[str, object], detail_metrics: List[str]) -> None:
    """
    Fill original metric columns with predefined missing penalty values.
    Only metrics defined in MISSING_FILLS are filled.
    """
    for m in detail_metrics:
        if m in MISSING_FILLS:
            rec[m] = MISSING_FILLS[m]


# ============================================================
# Main task runner
# ============================================================

def run_task(
    input_root: str,
    gt_dir: str,
    output_dir: str,
    example_dir: str,
    task: str,
    flowvn_dir: Optional[str] = None,
) -> None:
    cfg = TASK_CONFIG[task]
    dir_name = cfg["dir_name"]
    detail_metrics = cfg["detail_metrics"]
    summary_metrics = cfg["summary_metrics"]
    summary_level = cfg["summary_level"]
    needs_flowvn = cfg["needs_flowvn"]
    group_col = summary_level

    if needs_flowvn and not flowvn_dir:
        raise RuntimeError(f"Task {task} requires --flowvn_dir")

    # warm-up cache
    get_cached_ssim3d()

    settype = "ValidationSet"
    ex_set_root = os.path.join(example_dir, dir_name, settype)
    if not os.path.isdir(ex_set_root):
        raise RuntimeError(f"example_dir does not contain: {ex_set_root}")

    ex_rels = sorted(
        norm_relpath(p, example_dir)
        for p in list_files(ex_set_root)
        if p.endswith(".npz")
        and os.path.basename(p).startswith("img_ktGaussian")
    )

    samples: List[SampleID] = []
    expected_per_group: Dict[str, int] = {}

    for rel in ex_rels:
        sid = extract_sample_id_from_relpath(rel, dir_name=dir_name)
        if sid is None:
            raise RuntimeError(f"Example contains unexpected path: {rel}")
        samples.append(sid)
        gk = sid.center if group_col == "Center" else sid.anatomy
        expected_per_group[gk] = expected_per_group.get(gk, 0) + 1

    total_expected = sum(expected_per_group.values())

    records: List[Dict] = []
    t0 = time.time()

    for sid in samples:
        pred_path = os.path.join(input_root, *sid.rel_pred_path.split("/"))

        gt_base = os.path.join(
            gt_dir, dir_name, sid.settype,
            sid.anatomy, sid.center, sid.vendor, sid.patient,
        )
        gt_path = os.path.join(gt_base, "img_gt.npz")
        seg_path = os.path.join(gt_base, "segmask.mat")

        rec: Dict[str, object] = {
            "Task": task,
            "SetType": sid.settype,
            "Anatomy": sid.anatomy,
            "Center": sid.center,
            "Vendor": sid.vendor,
            "Patient": sid.patient,
            "R": sid.r,
            "PredRelPath": sid.rel_pred_path,
        }
        for m in detail_metrics:
            rec[m] = np.nan
        rec["Comments"] = ""

        missing_files: List[str] = []
        if not os.path.exists(pred_path):
            missing_files.append("pred")
        if not os.path.exists(gt_path):
            missing_files.append("gt")
        if not os.path.exists(seg_path):
            missing_files.append("segmask")

        fvn_path: Optional[str] = None
        if needs_flowvn:
            fvn_path = os.path.join(
                flowvn_dir, dir_name, sid.settype,
                sid.anatomy, sid.center, sid.vendor, sid.patient,
                f"img_ktGaussian{sid.r}.npz",
            )
            if not os.path.exists(fvn_path):
                missing_files.append("flowvn")

        if missing_files:
            rec["Comments"] = "data missing"
            fill_metrics_with_missing_values(rec, detail_metrics)
            records.append(rec)
            continue

        gt_arr = pred_arr = seg_arr = fvn_arr = corr_maps = None
        try:
            gt_arr = load_npz_dense(gt_path)
            pred_arr = load_npz_dense(pred_path)
            seg_arr = load_segmask_mat(seg_path)

            if gt_arr.shape != pred_arr.shape:
                rec["Comments"] = f"shape error: gt{gt_arr.shape} vs pred{pred_arr.shape}"
                fill_metrics_with_missing_values(rec, detail_metrics)
            else:
                corr_maps = get_or_create_corrmap(gt_arr, seg_arr, gt_base)

                if needs_flowvn:
                    fvn_arr = load_npz_dense(fvn_path)
                    if gt_arr.shape != fvn_arr.shape:
                        rec["Comments"] = f"shape error: gt{gt_arr.shape} vs flowvn{fvn_arr.shape}"
                        fill_metrics_with_missing_values(rec, detail_metrics)
                    else:
                        rec.update(
                            compute_metrics_r2(
                                gt_arr, pred_arr, fvn_arr, seg_arr, corr_maps=corr_maps
                            )
                        )
                else:
                    rec.update(
                        compute_metrics_standard(
                            gt_arr, pred_arr, seg_arr, corr_maps=corr_maps
                        )
                    )

        except Exception as e:
            rec["Comments"] = f"Exception: {type(e).__name__}: {e}"
            fill_metrics_with_missing_values(rec, detail_metrics)

        records.append(rec)

        del gt_arr, pred_arr, seg_arr, fvn_arr, corr_maps
        gc.collect()

    elapsed = time.time() - t0

    df = pd.DataFrame(records)

    if "ComplexDiffErr_FlowVN" in df.columns:
        df.drop(columns=["ComplexDiffErr_FlowVN"], inplace=True)

    if "Comments" not in df.columns:
        df["Comments"] = ""

    valid_mask = (df["Comments"] == "") | df["Comments"].isna()
    valid_df = df[valid_mask].copy()

    summary = OrderedDict()
    summary["submission_status"] = "SCORED"
    summary["Num_Files"] = f"{int(valid_mask.sum())}/{total_expected}"

    all_groups = sorted(expected_per_group.keys())

    for gval in all_groups:
        expected = expected_per_group[gval]
        gv = valid_df[valid_df[group_col] == gval]
        submitted = len(gv)

        summary[f"{gval}_Num"] = f"{submitted}/{expected}"

        for met in summary_metrics:
            if met in gv.columns:
                vals = gv[met].dropna().astype(float).tolist()
            else:
                vals = []
            fill = MISSING_FILLS.get(met, 0.0)
            adj = adjusted_mean(vals, expected, fill)
            summary[f"{gval}_{met}_adj"] = round(adj, 6)

        if task == "R2":
            if "BetterThanFlowVN" in gv.columns and submitted > 0:
                better = int((gv["BetterThanFlowVN"] == "Yes").sum())
            else:
                better = 0
            summary[f"{gval}_BetterThanFlowVN"] = f"{better}/{expected}"

    total_submitted = int(valid_mask.sum())

    for met in summary_metrics:
        if met in valid_df.columns:
            all_vals = valid_df[met].dropna().astype(float).tolist()
        else:
            all_vals = []
        fill = MISSING_FILLS.get(met, 0.0)
        adj = adjusted_mean(all_vals, total_expected, fill)
        summary[f"{met}_adj"] = round(adj, 6)

    if task == "R2":
        if "BetterThanFlowVN" in valid_df.columns and total_submitted > 0:
            total_better = int((valid_df["BetterThanFlowVN"] == "Yes").sum())
        else:
            total_better = 0
        summary["BetterThanFlowVN"] = f"{total_better}/{total_expected}"

    result_root = os.path.join(output_dir, "Result")
    os.makedirs(result_root, exist_ok=True)

    json_summary = OrderedDict(summary)

    json_path = os.path.join(result_root, f"results_{task}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)

    excel_path = os.path.join(result_root, f"result_{task}.xlsx")
    summary_df = pd.DataFrame(
        [{"Category": k, "Value": v} for k, v in summary.items()]
    )
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="results", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"Task {task} completed in {elapsed:.1f}s")
    print(f"  JSON:  {json_path}")
    print(f"  Excel: {excel_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="4D Flow MRI Reconstruction Challenge — Validation Scorer"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Submission directory or archive (.zip/.tar/.tgz/.tar.gz/.gz/.rar)",
    )
    parser.add_argument(
        "-t", "--task", required=True,
        choices=["R1", "R2", "S1", "S2"],
        help="Task to score: R1, R2, S1, or S2",
    )
    parser.add_argument(
        "-g", "--gt_dir", required=True,
        help="Ground-truth root directory",
    )
    parser.add_argument(
        "-x", "--example_dir", required=True,
        help="Example directory used as structure reference",
    )
    parser.add_argument(
        "-o", "--output", default="./",
        help="Output directory (Result/ will be created here)",
    )
    parser.add_argument(
        "--flowvn_dir", default=None,
        help="FlowVN baseline directory (required for task R2)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    temp_dir_to_cleanup: Optional[str] = None
    try:
        extracted_root, temp_dir_to_cleanup = extract_to_temp(
            args.input, output_dir
        )
        run_task(
            input_root=extracted_root,
            gt_dir=os.path.abspath(args.gt_dir),
            output_dir=output_dir,
            example_dir=os.path.abspath(args.example_dir),
            task=args.task,
            flowvn_dir=(
                os.path.abspath(args.flowvn_dir) if args.flowvn_dir else None
            ),
        )
    finally:
        if temp_dir_to_cleanup is not None:
            shutil.rmtree(temp_dir_to_cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()



"""
Examples:

# Task R1
python Val_Score2026_v3.py \\
    -i //mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_FlowVN/Submission_TaskR1_ValidationSet.zip \\
    -t R1 \\
    -g /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/ \\
    -x /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_EMPTY \\
    -o ./

# Task R2 (needs --flowvn_dir)
python Val_Score2026_v3.py \\
    -i //mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_FlowVN/Submission_TaskR2_ValidationSet_bad.zip \\
    -t R2 \\
    -g /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/ \\
    -x /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_EMPTY \\
    -o ./ \\
    --flowvn_dir /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_FlowVN/

# Task S1
python Val_Score2026_v3.py \\
    -i //mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_FlowVN/Submission_TaskS1_ValidationSet.zip \\
    -t S1 \\
    -g /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/ \\
    -x /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_EMPTY \\
    -o ./

# Task S2
python Val_Score2026_v3.py \\
    -i //mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_FlowVN/Submission_TaskS2_ValidationSet.zip \\
    -t S2 \\
    -g /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/ \\
    -x /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_EMPTY \\
    -o ./
"""