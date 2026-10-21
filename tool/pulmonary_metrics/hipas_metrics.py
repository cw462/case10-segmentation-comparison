"""Transparent hard-label metrics associated with HiPaS (Nature Commun. 2025).

This is an independent implementation, not the authors' evaluation program.
In particular BC and SL are explicitly native-grid, Lee-skeleton proxies.
No resampling, A/V repair, component removal, or hidden smoothing is performed.

Reference: https://doi.org/10.1038/s41467-025-56505-6
Methods > Statistical analyses (MCS, BC/SL), Results > Fig. 2 (evaluation).
Requires numpy, scipy, scikit-image, and surface-distance 0.1.
Array spacing follows axis 0, 1, 2;
array axes are not assumed to be anatomical x/y/z. Values are fractions, not %.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata
from skimage.morphology import skeletonize
import surface_distance as sd


def _validate_inputs(pred, gt, spacing, artery_label, vein_label, roi):
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape != gt.shape:
        raise ValueError("pred and gt must be 3-D arrays with identical shapes")
    if pred.size == 0:
        raise ValueError("pred and gt must not have empty dimensions")
    for name, arr in (("pred", pred), ("gt", gt)):
        if arr.dtype.kind not in "biuf":
            raise ValueError(f"{name} must contain finite integer-valued labels")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or infinity")
        if (arr < 0).any():
            raise ValueError(f"{name} contains negative labels")
        if arr.dtype.kind == "f" and not np.equal(arr, np.floor(arr)).all():
            raise ValueError(f"{name} contains non-integer labels/probabilities")
    for name, label in (("artery_label", artery_label), ("vein_label", vein_label)):
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) or label <= 0:
            raise ValueError(f"{name} must be an explicit positive integer")
    if artery_label == vein_label:
        raise ValueError("artery_label and vein_label must differ")
    try:
        spacing = np.asarray(spacing, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("spacing must be three positive finite millimetre values") from exc
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing must be three positive finite millimetre values")
    if roi is not None:
        roi = np.asarray(roi)
        if roi.shape != pred.shape or roi.dtype.kind not in "biuf":
            raise ValueError("roi must be a binary array with the same shape as pred")
        if not np.isfinite(roi).all() or not np.logical_or(roi == 0, roi == 1).all():
            raise ValueError("roi must contain only 0 and 1")
        roi = roi.astype(bool, copy=False)
        if not roi.any():
            raise ValueError("roi must contain at least one evaluated voxel")
    return pred, gt, tuple(float(x) for x in spacing), roi


def _bbox(mask):
    """Tight nonempty-mask box, without constructing a list of all coordinates."""
    slices = []
    for axis in range(3):
        active = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        slices.append(slice(int(active[0]), int(active[-1]) + 1))
    return tuple(slices)


def _hd95(pred_mask, gt_mask, spacing):
    """Area-weighted directional surface HD95, maximum of both directions."""
    p_any, g_any = bool(pred_mask.any()), bool(gt_mask.any())
    if not p_any or not g_any:
        reason = "undefined_both_empty" if not p_any and not g_any else "undefined_one_empty"
        return None, reason
    distances = sd.compute_surface_distances(gt_mask, pred_mask, spacing)
    return float(sd.compute_robust_hausdorff(distances, 95)), "ok_area_weighted_directional95_max"


def _native_skeleton_proxy(mask):
    """Return branch-junction clusters and skeleton voxel count, not mm length.

    Degree is measured using all 26 neighbours. Adjacent degree>=3 skeleton
    voxels are grouped into one 26-connected junction cluster. Digital diagonal
    contacts, loops and adjacent junctions make this an imperfect BC proxy.
    """
    if not mask.any():
        return 0, 0
    cropped = np.pad(mask[_bbox(mask)], 1, constant_values=False)
    skel = np.asarray(skeletonize(cropped, method="lee"), dtype=bool)
    sl = int(np.count_nonzero(skel))
    neighbours = np.ones((3, 3, 3), dtype=np.uint8)
    neighbours[1, 1, 1] = 0
    degree = ndi.convolve(skel.astype(np.uint8), neighbours, mode="constant", cval=0)
    candidates = skel & (degree >= 3)
    _, bc = ndi.label(candidates, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return int(bc), sl


def _validate_probability(probability, shape, name):
    if probability is None:
        return None
    arr = np.asarray(probability)
    if arr.shape != shape or arr.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a numeric probability array on the label grid")
    if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return arr


def _roc_auc(reference_mask, probability, roi):
    """Mann-Whitney ROC AUC using average ranks (ties receive half credit)."""
    if probability is None:
        return None, "not_computed_probability_not_supplied"
    truth = reference_mask.ravel() if roi is None else reference_mask[roi]
    scores = probability.ravel() if roi is None else probability[roi]
    n_positive = int(np.count_nonzero(truth))
    n_negative = int(truth.size - n_positive)
    if not n_positive:
        return None, "undefined_no_reference_positive"
    if not n_negative:
        return None, "undefined_no_reference_negative"
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(ranks[truth].sum(dtype=np.float64))
    u = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(u / (n_positive * n_negative)), "ok_supplied_probability_scores"


def evaluate_hipas(
    pred, gt, spacing, artery_label, vein_label, roi=None,
    *, artery_probability=None, vein_probability=None,
) -> dict[str, Any]:
    """Evaluate mutually exclusive arterial/venous hard labels on one shared grid.

    Parameters
    ----------
    pred, gt : array-like, same 3-D shape
        Hard labels; every label other than artery_label/vein_label is treated
        as nonvascular background. Callers must verify physical registration
        and identical geometry: this array API cannot inspect image affines.
    spacing : sequence of three positive floats, REQUIRED
        Millimetres in array-axis order (axis 0, 1, 2). Orthogonal voxel axes
        are required; sheared image geometry must be handled by the caller.
    artery_label, vein_label : distinct positive integers
    roi : optional binary array of the same shape
        Restricts voxel confusion counts, MCS, and AUC ONLY. HD95 and skeleton
        proxies always evaluate full masks, avoiding artificial cut surfaces.
    artery_probability, vein_probability : optional same-grid arrays in [0, 1]
        Independently supplied class probability maps for one-vs-rest ROC AUC.
        No probabilities are derived from the hard segmentation. Ranking uses
        all evaluated voxels, so large volumes can require substantial memory.

    Returns
    -------
    dict with metrics (flat float/None), status (one entry per metric), notes,
    and protocol. None denotes an undefined denominator/empty-surface case.
    No paper-exact BC/SL or exact author ROC-pooling protocol is claimed.
    """
    pred, gt, spacing, roi = _validate_inputs(
        pred, gt, spacing, artery_label, vein_label, roi
    )
    probabilities = {
        "artery": _validate_probability(artery_probability, pred.shape, "artery_probability"),
        "vein": _validate_probability(vein_probability, pred.shape, "vein_probability"),
    }
    metrics, status = {}, {}
    notes = [
        "Independent implementation; source-paper evaluator equivalence is not established.",
        "HD95 follows the package-wide surface-distance 0.1 protocol: surfel-area-weighted directional 95th percentiles, then their maximum.",
        "BC_proxy and SL_voxels_proxy use native-grid Lee skeletonization; no resampling is performed.",
        "The paper standardizes to 334/512 x 334/512 x 1.00 mm and uses an octree-based skeleton method; these proxies are not paper-exact BC/SL.",
        "SL_voxels_proxy counts skeleton voxels, not physical millimetres; compare only on explicitly matched grids.",
        "BC/SL ratios are abundance ratios, not matched-branch recall; excess branches or length do not establish better segmentation.",
        "MCS uses the printed numerator divided by predicted-plus-reference vascular voxel counts; no factor of two is added. Complete A/V swapping gives 0.5.",
        "AUC uses only separately supplied probability maps, one-vs-rest per case; no AUC is inferred from hard labels.",
        "The paper's expert reader scores require human review and are not automated by this evaluator.",
    ]
    if roi is not None:
        notes.append("ROI affects DSC/SEN/SPE/FPR/MCS/AUC only; HD95 and skeleton proxies still evaluate full-volume masks.")
    for name, label in (("artery", artery_label), ("vein", vein_label)):
        p, g = pred == label, gt == label
        pc, gc = (p, g) if roi is None else (p[roi], g[roi])
        tp = int(np.count_nonzero(pc & gc))
        fp = int(np.count_nonzero(pc & ~gc))
        fn = int(np.count_nonzero(~pc & gc))
        tn = int(pc.size - tp - fp - fn)
        denominator = 2 * tp + fp + fn
        key = f"{name}_DSC"
        metrics[key] = float(2 * tp / denominator) if denominator else None
        status[key] = "ok" if denominator else "undefined_both_empty"
        for metric_name, numerator, denominator, absent in (
            ("SEN", tp, tp + fn, "undefined_no_reference_positive"),
            ("SPE", tn, tn + fp, "undefined_no_reference_negative"),
            ("FPR", fp, tn + fp, "undefined_no_reference_negative"),
        ):
            key = f"{name}_{metric_name}"
            metrics[key] = float(numerator / denominator) if denominator else None
            status[key] = "ok" if denominator else absent
        key = f"{name}_HD95_mm"
        metrics[key], status[key] = _hd95(p, g, spacing)
        key = f"{name}_AUC"
        metrics[key], status[key] = _roc_auc(g, probabilities[name], roi)
        bc_p, sl_p = _native_skeleton_proxy(p)
        bc_g, sl_g = _native_skeleton_proxy(g)
        for measure, value_p, value_g in (
            ("BC_proxy", bc_p, bc_g),
            ("SL_voxels_proxy", sl_p, sl_g),
        ):
            for suffix, value in (("pred", value_p), ("gt", value_g)):
                key = f"{name}_{measure}_{suffix}"
                metrics[key] = float(value)
                status[key] = "native_grid_proxy_not_paper_exact"
            key = f"{name}_{measure}_ratio"
            metrics[key] = float(value_p / value_g) if value_g else None
            status[key] = "native_grid_proxy_abundance_ratio" if value_g else "undefined_reference_proxy_zero"
    pa, pv, ga, gv = pred == artery_label, pred == vein_label, gt == artery_label, gt == vein_label
    if roi is not None:
        pa, pv, ga, gv = pa[roi], pv[roi], ga[roi], gv[roi]
    swaps = int(np.count_nonzero(pa & gv) + np.count_nonzero(pv & ga))
    vascular_sum = int(np.count_nonzero(pa | pv) + np.count_nonzero(ga | gv))
    metrics["MCS"] = float(swaps / vascular_sum) if vascular_sum else None
    status["MCS"] = "paper_printed_formula_count_interpretation" if vascular_sum else "undefined_no_vascular_voxels"
    return {
        "metrics": metrics,
        "status": status,
        "notes": notes,
        "protocol": {
            "source": "https://doi.org/10.1038/s41467-025-56505-6",
            "source_location": "Methods: Statistical analyses; Results: Fig. 2",
            "implementation": "independent; hard-label metrics plus explicitly named native-grid skeleton proxies",
            "spacing_mm_array_axis_order": list(spacing),
            "geometry_requirement": "same shape, shared aligned orthogonal voxel grid; affine agreement must be checked by caller",
            "artery_label": int(artery_label),
            "vein_label": int(vein_label),
            "other_labels": "treated as nonvascular background",
            "fraction_scale": "0 to 1, not percent; MCS printed-formula interpretation is 0 to 0.5",
            "roi_scope": "voxel_confusion_MCS_and_AUC_only" if roi is not None else "full_volume",
            "roi_voxel_count": int(np.count_nonzero(roi)) if roi is not None else int(pred.size),
            "surface_and_skeleton_scope": "full_volume_even_when_roi_supplied",
            "FPR": "FP/(FP+TN) = 1-SPE; negatives are one-vs-rest within the confusion-count scope",
            "HD95": "surface-distance 0.1; surfel-area-weighted 95th percentile in each direction, then maximum, in mm",
            "HD95_caveat": "unified package convention; the paper's exact surface implementation was not verified",
            "empty_DSC": "both empty: None; exactly one empty: 0",
            "empty_HD95": "None if either surface is empty",
            "undefined_denominator": "None, with per-metric status; not replaced by zero or infinity",
            "MCS": "(|P_A intersect G_V| + |P_V intersect G_A|) / (|P_A union P_V| + |G_A union G_V|)",
            "MCS_caveat": "literal printed P+T denominator interpreted as total predicted plus reference vascular counts; official evaluator not verified",
            "skeletonization": "scikit-image skeletonize(method='lee'), no pruning or component removal",
            "BC_proxy": "26-connected component count of skeleton voxels with 26-neighbour degree >=3",
            "SL_voxels_proxy": "number of foreground skeleton voxels, not length in mm",
            "proxy_ratio": "prediction count / reference count; undefined when reference count is zero; not branch matching recall",
            "resampling": "none; original paper's standardized grid is not silently applied",
            "paper_standard_spacing_mm": [334.0 / 512.0, 334.0 / 512.0, 1.0],
            "AUC": "per-case one-vs-rest Mann-Whitney U/(n_positive*n_negative), scipy rankdata average ties; no hard-label substitution",
            "AUC_probability_supplied": {name: value is not None for name, value in probabilities.items()},
            "AUC_pooling_caveat": "author cohort-level ROC pooling/aggregation was not verified; this implementation returns per-case AUC",
            "reader_scores": "human-assessed outcomes, not computed automatically",
        },
    }
