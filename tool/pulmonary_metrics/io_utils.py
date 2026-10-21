"""Physical-grid validation and safe 3D hard-label loading."""
from pathlib import Path
import numpy as np
from vesmamba_metrics import validate_spacing


def validate_labels(a, name='mask'):
    a = np.asarray(a)
    if a.ndim != 3 or not a.size or any(n == 0 for n in a.shape):
        raise ValueError(f'{name}: expected nonempty 3D labels')
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == np.bool_):
        raise ValueError(f'{name}: expected numeric labels')
    if np.iscomplexobj(a) or not np.isfinite(a).all() or (a < 0).any() or not np.equal(a, np.floor(a)).all():
        raise ValueError(f'{name}: expected finite nonnegative integer labels; probabilities/ignore labels unsupported')
    return a


def load_volume(path, assume_mm=False, kind='labels'):
    path = Path(path)
    affine = spacing = None
    notes = []
    if path.name.lower().endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        img = nib.load(str(path))
        a = np.asanyarray(img.dataobj)
        affine = np.asarray(img.affine, dtype=float).copy()
        unit = img.header.get_xyzt_units()[0]
        if unit == 'unknown' and not assume_mm:
            raise ValueError(f'{path.name}: NIfTI spatial units unknown; verify units then explicitly use --assume-mm if appropriate')
        if unit == 'unknown':
            notes.append(f'{path.name}: unknown NIfTI units explicitly interpreted as mm')
        if unit not in ('mm', 'meter', 'micron', 'unknown'):
            raise ValueError(f'{path.name}: unsupported spatial unit {unit}')
        affine[:3, :] *= {'mm': 1, 'meter': 1000, 'micron': .001, 'unknown': 1}[unit]
        matrix = affine[:3, :3]
        spacing = validate_spacing(np.linalg.norm(matrix, axis=0))
        direction = matrix / spacing
        if not np.isfinite(affine).all() or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
            raise ValueError(f'{path.name}: nonfinite or sheared affine unsupported')
    elif path.suffix.lower() == '.npy':
        a = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as z:
            key = 'mask' if 'mask' in z.files else z.files[0] if len(z.files) == 1 else None
            if key is None:
                raise ValueError('NPZ must contain key mask or exactly one array')
            a = z[key]
    else:
        raise ValueError(f'unsupported format: {path}; use NIfTI/NPY/NPZ')
    if kind == 'labels':
        a = validate_labels(a, str(path))
    elif kind == 'probability':
        a = np.asarray(a)
        if a.ndim != 3 or not a.size or not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
            raise ValueError('probability must be a nonempty numeric 3D array')
        if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
            raise ValueError('probabilities must be finite in [0,1], not logits')
    else:
        raise ValueError('unknown volume kind')
    return {'data': a, 'affine': affine, 'spacing': spacing, 'notes': notes, 'path': str(path)}


def check_grids(volumes, spacing=None):
    values = list(volumes.values())
    shapes = {v['data'].shape for v in values}
    if len(shapes) != 1:
        raise ValueError('all masks must have identical shapes; no automatic resampling')
    affines = [v['affine'] for v in values if v['affine'] is not None]
    if affines and len(affines) != len(values):
        raise ValueError('do not mix NIfTI with NumPy masks; NumPy has no verifiable affine')
    s = validate_spacing(spacing) if spacing is not None else values[0]['spacing']
    if s is None:
        raise ValueError('NumPy masks require --spacing S0 S1 S2, in mm and array-axis order')
    for v in values:
        if v['spacing'] is not None and not np.allclose(v['spacing'], s, rtol=1e-5, atol=1e-5):
            raise ValueError('voxel spacing mismatch; --spacing cannot override a NIfTI grid')
        if affines and not np.allclose(v['affine'], affines[0], rtol=0, atol=1e-4):
            raise ValueError('NIfTI affines differ: origin/orientation/spacing must match')
    return validate_spacing(s)
