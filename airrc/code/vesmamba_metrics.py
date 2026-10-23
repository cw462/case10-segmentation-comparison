"""Four VesMamba metric names with an explicit, independent evaluation protocol.
This is not the authors' evaluation code. Ratios are in [0,1], distances in mm.
"""
from __future__ import annotations
import numpy as np
from skimage.morphology import skeletonize
from surface_distance import metrics as sd

METRICS = ('Dice', 'clDice', 'HD95_mm', 'NSD')


def validate_spacing(spacing):
    s = np.asarray(spacing, dtype=float)
    if s.shape != (3,) or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError('spacing must be three finite positive values in NumPy axis order, in mm')
    return tuple(float(x) for x in s)


def validate_binary(mask, name):
    a = np.asarray(mask)
    if a.ndim != 3 or not a.size or any(n == 0 for n in a.shape):
        raise ValueError(f'{name}: expected a nonempty 3D array')
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == np.bool_):
        raise ValueError(f'{name}: expected numeric or Boolean data')
    if np.iscomplexobj(a) or not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
        raise ValueError(f'{name}: binary hard mask must contain only 0/1; select a class explicitly')
    return a.astype(bool, copy=False)


def _crop_with_padding(p, g):
    union = p | g
    slices = []
    for axis in range(3):
        indices = np.flatnonzero(np.any(union, axis=tuple(i for i in range(3) if i != axis)))
        slices.append(slice(int(indices[0]), int(indices[-1]) + 1))
    region = tuple(slices)
    # Only remove background outside the union bbox. Both masks receive the same zero border.
    return np.pad(p[region], 1), np.pad(g[region], 1)


def evaluate_binary(pred, gt, spacing, nsd_tolerance_mm):
    """Return four values and per-metric status; never hide undefined values as NaN.

    Dice: 2|P&G|/(|P|+|G|).
    clDice: harmonic mean of skeleton precision and skeleton sensitivity, Lee 3D.
    HD95: max of two directional surfel-area-weighted 95th percentiles, in mm.
    NSD: sum of in-tolerance surface areas / sum of both surface areas.
    No resampling, connected-component filtering, smoothing or branch pruning.
    """
    spacing = validate_spacing(spacing)
    if not np.isscalar(nsd_tolerance_mm) or not np.isfinite(nsd_tolerance_mm) or nsd_tolerance_mm < 0:
        raise ValueError('nsd_tolerance_mm must be finite and >=0')
    p, g = validate_binary(pred, 'pred'), validate_binary(gt, 'gt')
    if p.shape != g.shape:
        raise ValueError('pred and gt must have identical shapes and physical grids')
    n_p, n_g = int(p.sum()), int(g.sum())
    result = dict.fromkeys(METRICS, None)
    status = dict.fromkeys(METRICS, 'ok')
    result['status'] = status
    if n_p == 0 and n_g == 0:
        status.update(dict.fromkeys(METRICS, 'both masks empty; undefined, do not count as a perfect case'))
        return result
    result['Dice'] = 2 * int((p & g).sum()) / (n_p + n_g)
    if n_p == 0 or n_g == 0:
        result['clDice'] = 0.0
        result['NSD'] = 0.0
        status['clDice'] = 'one mask empty; set to 0 by declared convention'
        status['NSD'] = 'one mask empty; set to 0 by declared convention'
        status['HD95_mm'] = 'one surface empty; no finite distance, recorded as null (failure, not zero)'
        return result
    p, g = _crop_with_padding(p, g)
    sp = skeletonize(p, method='lee').astype(bool)
    sg = skeletonize(g, method='lee').astype(bool)
    if sp.any() and sg.any():
        tprec = int((sp & g).sum()) / int(sp.sum())
        tsens = int((sg & p).sum()) / int(sg.sum())
        result['clDice'] = 0.0 if tprec + tsens == 0 else 2 * tprec * tsens / (tprec + tsens)
    else:
        status['clDice'] = 'Lee skeleton empty despite nonempty mask; inspect geometry, do not silently drop case'
    distances = sd.compute_surface_distances(g, p, spacing)
    result['HD95_mm'] = float(sd.compute_robust_hausdorff(distances, percent=95))
    result['NSD'] = float(sd.compute_surface_dice_at_tolerance(distances, float(nsd_tolerance_mm)))
    return result


def evaluate_vesmamba(pred, gt, spacing, nsd_tolerance_mm, labels=None):
    """Dataset label IDs must match between prediction and GT; no implicit A/V union."""
    metrics, statuses, notes = {}, {}, []
    if labels is None:
        groups = [('binary', pred, gt)]
    else:
        from io_utils import validate_labels
        pred, gt = validate_labels(pred, 'pred'), validate_labels(gt, 'gt')
        if pred.shape != gt.shape:
            raise ValueError('pred and gt shapes differ')
        if not labels or any(int(k) != k or k <= 0 for k in labels) or len(set(labels)) != len(labels):
            raise ValueError('labels must be distinct positive integers')
        groups = [(f'label_{label}', np.asarray(pred) == label, np.asarray(gt) == label) for label in labels]
    for name, p, g in groups:
        score = evaluate_binary(p, g, spacing, nsd_tolerance_mm)
        for metric in METRICS:
            key = f'{name}.{metric}'
            metrics[key], statuses[key] = score[metric], score['status'][metric]
        if np.array_equal(p, g):
            notes.append(f'{name}: prediction and reference are identical; check that the evaluation is not against itself.')
    return {'metrics': metrics, 'status': statuses, 'notes': notes,
            'protocol': {'implementation': 'independent fixed protocol, not VesMamba authors code',
            'spacing_mm_array_axes': list(validate_spacing(spacing)),
            'NSD_tolerance_mm': float(nsd_tolerance_mm),
            'NSD': 'surfel-area weighted surface Dice, inclusive distance <= tolerance',
            'HD95': 'max of directional area-weighted 95th percentiles, surface-distance 0.1, mm',
            'clDice': '3D Lee skeleton, voxel-count precision/sensitivity, no pruning',
            'postprocessing': 'none; union bbox crop removes only shared background, then zero padding',
            'scale': 'Dice/clDice/NSD fractions, HD95 mm',
            'empty_policy': 'both empty=null; one empty overlap metrics=0, HD95=null with failure status'}}
