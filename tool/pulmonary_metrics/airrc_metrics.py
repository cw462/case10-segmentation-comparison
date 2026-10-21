"""AirRC voxel metrics and an explicitly independent branch protocol.

The AirRC paper reports DSC, Precision and Recall for four structures, and
BDR/DLR for airway lumen only. No claim of exact author-code reproduction is
made for branch matching. See airrc_notes.txt before reporting branch scores.
"""

from itertools import product
from numbers import Integral
from collections.abc import Mapping

import numpy as np

PAPER_URL = "https://www.nature.com/articles/s41597-025-06074-6"
AUTHOR_REPOSITORY = "https://github.com/b-niu/AirRC"
BRANCH_PROTOCOL = "fixed_gt_branch_26edge_mm_strict_gt80_v1_independent"


def _label_array(value, name):
    a = np.asarray(value)
    if a.ndim != 3 or a.size == 0:
        raise ValueError(f"{name} must be a nonempty 3D label array")
    if a.dtype.kind not in "buif" or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain finite nonnegative integer labels")
    if np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError(f"{name} must contain nonnegative integer labels")
    return a


def _branch_edges(coords, spacing, branch_id):
    """Validate a simple 26-neighbour path; return unique physical edges.

    Rejecting junctions/loops prevents a dense branch mask or ambiguous diagonal
    shortcut graph from silently becoming a biased length measurement.
    """
    vertices = {tuple(int(v) for v in p) for p in coords}
    if len(vertices) < 2:
        raise ValueError(f"branch {branch_id}: need at least two centerline voxels")
    offsets = [d for d in product((-1, 0, 1), repeat=3) if d > (0, 0, 0)]
    neighbours = {p: [] for p in vertices}
    edges = []
    for p in vertices:
        for d in offsets:
            q = tuple(p[i] + d[i] for i in range(3))
            if q in vertices:
                length = float(np.linalg.norm(np.asarray(d) * spacing))
                edges.append((p, q, length))
                neighbours[p].append(q)
                neighbours[q].append(p)
    unseen = set(vertices)
    stack = [next(iter(vertices))]
    while stack:
        p = stack.pop()
        if p in unseen:
            unseen.remove(p)
            stack.extend(neighbours[p])
    degrees = [len(neighbours[p]) for p in vertices]
    if unseen or max(degrees) > 2 or degrees.count(1) != 2:
        raise ValueError(
            f"branch {branch_id}: ID must form one simple 26-neighbour path "
            "with two endpoints, no loops/junctions/diagonal shortcuts; "
            "supply a reviewed centerline branch map, not a thick region mask"
        )
    return edges


def evaluate_airrc(pred, gt, spacing, labels, airway_label=None, branch_labels=None):
    """Evaluate aligned 3D integer masks.

    Parameters
    ----------
    pred, gt : array-like, same 3D shape
        Aligned discrete label masks. The caller must check image affine/grid.
    spacing : sequence of three positive floats
        Millimetres, in the same axis order as the arrays (not assumed xyz).
    labels : mapping[str, int]
        Explicit class name -> positive label ID. Background is 0. A subset
        may be evaluated; other labels remain valid negatives for each class.
    airway_label : int or str, optional
        Airway LUMEN ID (or its key in labels), never the airway wall label.
    branch_labels : integer array, optional
        Same grid. 0 outside the reviewed reference centerline; each positive
        ID is one GT airway branch. See notes for branch/junction convention.

    Returns a JSON-safe dict: metrics (flat floats/None), status (flat strings),
    notes, protocol. BDR/DLR names include "independent" deliberately.
    """
    p = _label_array(pred, "pred")
    g = _label_array(gt, "gt")
    if p.shape != g.shape:
        raise ValueError("pred and gt shapes differ; resample/align before evaluation")
    s = np.asarray(spacing, dtype=float)
    if s.shape != (3,) or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("spacing must have three finite positive millimetre values")
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError("labels must be a nonempty name-to-positive-ID mapping")
    for name, label_id in labels.items():
        if not isinstance(name, str) or not name.strip() or "." in name:
            raise ValueError("class names must be nonempty strings without '.'")
        if isinstance(label_id, bool) or not isinstance(label_id, Integral) or label_id <= 0:
            raise ValueError("class IDs must be positive integers; background 0 is excluded")
    if len(set(labels.values())) != len(labels):
        raise ValueError("class IDs must be unique")
    label_map = {name: int(v) for name, v in labels.items()}
    metrics, status = {}, {}
    notes = [
        "DSC/Precision/Recall use standard unsmoothed one-versus-rest voxel counts.",
        "FPR is supplementary: the AirRC paper calculated it but omitted it from result tables.",
        "No uncertainty/ignore mask is supported. Do not turn uncertain or unannotated voxels into background.",
        "Array inputs do not retain image affine. Caller must verify identical orientation, origin and grid.",
    ]
    protocol = {
        "name": "AirRC_independent_evaluation_v1",
        "paper": PAPER_URL,
        "author_repository": AUTHOR_REPOSITORY,
        "author_code_check": {
            "date": "2026-10-20",
            "inspected_files": ["config_loss.py", "pulmonary_structure_segmentation_tools.py", "train_stage1.py", "train_stage2.py"],
            "finding": "training validation uses MONAI DiceMetric; BDR/DLR final evaluation implementation not found in these files",
        },
        "scale": "ratios in [0,1], not percentages",
        "spacing_mm_array_axes": s.tolist(),
        "labels": label_map,
        "zero_denominator": "None, with reason in status; never add epsilon",
        "FPR_definition": "FP/(FP+TN), negatives are all voxels not in this GT class",
        "case_aggregation": "per-case outputs; aggregation is caller responsibility",
        "branch_protocol": None,
    }

    def ratio(key, numerator, denominator, ok="ok_standard_definition"):
        metrics[key] = float(numerator / denominator) if denominator else None
        status[key] = ok if denominator else "undefined_zero_denominator"

    for name, label_id in label_map.items():
        pm, gm = p == label_id, g == label_id
        tp = int(np.count_nonzero(pm & gm))
        npred, ngt = int(pm.sum()), int(gm.sum())
        fp = npred - tp
        ratio(name + ".DSC", 2 * tp, npred + ngt)
        ratio(name + ".Precision", tp, npred)
        ratio(name + ".Recall", tp, ngt)
        ratio(name + ".FPR", fp, g.size - ngt, "ok_supplementary_FPR")

    if airway_label is None:
        if branch_labels is not None:
            raise ValueError("branch_labels requires an explicit airway LUMEN label")
        notes.append("Airway BDR/DLR not requested: pass airway_label to include their availability status.")
        return {"metrics": metrics, "status": status, "notes": notes, "protocol": protocol}

    if isinstance(airway_label, str):
        if airway_label not in label_map:
            raise ValueError("airway_label name is absent from labels")
        airway_name, airway_id = airway_label, label_map[airway_label]
    else:
        if isinstance(airway_label, bool) or not isinstance(airway_label, Integral):
            raise ValueError("airway_label must be a class name or integer ID")
        found = [name for name, val in label_map.items() if val == airway_label]
        if len(found) != 1:
            raise ValueError("airway_label ID must be included in labels")
        airway_name, airway_id = found[0], int(airway_label)
    bdr_key = airway_name + ".BDR_gt80_independent"
    dlr_key = airway_name + ".DLR_matched_gt_length_independent"
    protocol["branch_protocol"] = {
        "id": BRANCH_PROTOCOL,
        "author_exact_reproduction": False,
        "airway_name": airway_name,
        "airway_label": airway_id,
        "branch_definition": "fixed supplied reference branch IDs; no automatic branch extraction",
        "length": "sum of unique 26-neighbour intra-ID centerline edge lengths in mm",
        "edge_coverage": "edge is covered only if BOTH endpoints lie in predicted airway lumen",
        "detected": "covered_reference_branch_length / reference_branch_length > 0.80",
        "BDR": "number of detected supplied GT branches / number of supplied GT branches",
        "DLR": "FULL lengths of detected supplied GT branches / lengths of all supplied GT branches",
        "prediction_branch_matching": "none; uses prediction lumen mask, not predicted branch IDs",
        "reference_completeness": "caller must review completeness and freeze branch IDs before comparisons",
        "junction_convention": "cross-ID edges and unlabelled junction voxels excluded from length denominator",
    }
    notes.append(
        "AirRC does not fully specify branch extraction/matching in the inspected text. "
        "These branch metrics are an independent GT-centreline coverage protocol, not author-exact BDR/DLR."
    )
    if branch_labels is None:
        for key in (bdr_key, dlr_key):
            metrics[key] = None
            status[key] = "not_computed_requires_fixed_reference_branch_labels"
        return {"metrics": metrics, "status": status, "notes": notes, "protocol": protocol}
    branches = _label_array(branch_labels, "branch_labels")
    if branches.shape != g.shape:
        raise ValueError("branch_labels must share the GT shape and grid")
    if np.any((branches > 0) & (g != airway_id)):
        raise ValueError("every labelled centerline voxel must lie inside GT airway lumen")
    # Group centreline voxels once, avoiding one whole-CT scan per branch.
    all_coords = np.argwhere(branches > 0)
    all_ids = branches[tuple(all_coords.T)]
    order = np.argsort(all_ids, kind='stable')
    all_coords, all_ids = all_coords[order], all_ids[order]
    ids, starts = np.unique(all_ids, return_index=True)
    ends = np.r_[starts[1:], len(all_ids)]
    if not len(ids):
        for key in (bdr_key, dlr_key):
            metrics[key] = None
            status[key] = "undefined_no_reference_branches"
        return {"metrics": metrics, "status": status, "notes": notes, "protocol": protocol}
    pm = p == airway_id
    rows = []
    total_length = detected_length = 0.0
    detected_count = 0
    for branch_id, start, end in zip(ids, starts, ends):
        coords = all_coords[start:end]
        edges = _branch_edges(coords, s, int(branch_id))
        length = sum(e[2] for e in edges)
        covered = sum(e[2] for e in edges if pm[e[0]] and pm[e[1]])
        # Numerical equality to exactly 80% is not a strict-threshold success.
        delta = covered - 0.8 * length
        detected = delta > 32 * np.finfo(float).eps * length
        total_length += length
        if detected:
            detected_count += 1
            detected_length += length
        rows.append({"branch_id": int(branch_id), "voxel_count": int(len(coords)),
                     "reference_length_mm": float(length), "covered_length_mm": float(covered),
                     "coverage_fraction": float(covered / length), "detected_gt80": bool(detected)})
    ratio(bdr_key, detected_count, len(ids), "ok_independent_protocol_not_author_exact")
    ratio(dlr_key, detected_length, total_length, "ok_independent_protocol_not_author_exact")
    protocol["branch_protocol"]["branches"] = rows
    protocol["branch_protocol"]["total_reference_length_mm"] = float(total_length)
    protocol["branch_protocol"]["detected_full_reference_length_mm"] = float(detected_length)
    notes.append("Branch denominator includes only supplied IDs. Missing branch IDs bias BDR/DLR and must be corrected before use.")
    notes.append("DLR here counts whole detected reference branches; it is not raw skeleton coverage, and may exceed that coverage fraction.")
    return {"metrics": metrics, "status": status, "notes": notes, "protocol": protocol}
