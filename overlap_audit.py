"""Export native-grid overlap masks and a 26-neighbor region location table."""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs" / "after"
CHANGES = ROOT / "changes"
RESULTS = ROOT / "results"
PAIRS = (
    ("airway", "pulmonary_artery", 27, 11),
    ("pulmonary_artery", "pulmonary_vein", 836, 24),
)


def load_mask(name):
    image = nib.load(INPUTS / f"{name}.nii.gz")
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or int(data.min()) != 0 or int(data.max()) != 1:
        raise ValueError(f"Expected nonempty 3D binary mask: {name}")
    mask = data.astype(bool, copy=True)
    del data
    return image, mask


def main():
    CHANGES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for first, second, expected_voxels, expected_regions in PAIRS:
        first_image, first_mask = load_mask(first)
        second_image, second_mask = load_mask(second)
        if first_image.shape != second_image.shape or not np.allclose(
            first_image.affine, second_image.affine, rtol=0, atol=1e-6
        ):
            raise ValueError(f"Input grids differ: {first}, {second}")
        overlap = first_mask & second_mask
        del first_mask, second_mask
        voxel_count = int(np.count_nonzero(overlap))
        assert voxel_count == expected_voxels, (first, second, voxel_count)
        bbox = []
        for axis in range(3):
            indices = np.flatnonzero(overlap.any(axis=tuple(a for a in range(3) if a != axis)))
            bbox.append((int(indices[0]), int(indices[-1]) + 1))
        crop_slices = tuple(slice(low, high) for low, high in bbox)
        cropped = np.array(overlap[crop_slices], dtype=bool, order="C", copy=True)
        labels, region_count = ndimage.label(cropped, structure=np.ones((3, 3, 3), dtype=np.uint8))
        assert region_count == expected_regions, (first, second, region_count)
        sizes = np.bincount(labels[cropped], minlength=region_count + 1)
        region_slices = ndimage.find_objects(labels)
        voxel_volume_ml = float(abs(np.linalg.det(first_image.affine[:3, :3]))) / 1000.0
        pair = f"{first}__{second}"
        destination = CHANGES / f"after_{first}_{second}_overlap.nii.gz"
        header = first_image.header.copy()
        header.set_data_dtype(np.uint8)
        output_image = nib.Nifti1Image(overlap.astype(np.uint8), first_image.affine, header=header)
        qform, qcode = first_image.get_qform(coded=True)
        sform, scode = first_image.get_sform(coded=True)
        output_image.set_qform(qform, int(qcode))
        output_image.set_sform(sform, int(scode))
        output_image.header["descrip"] = b"Post-edit binary mask intersection; location aid, not a GT error mask"
        nib.save(output_image, destination)
        del output_image, overlap
        saved = nib.load(destination)
        assert saved.shape == first_image.shape
        assert np.array_equal(saved.affine, first_image.affine)
        assert int(np.count_nonzero(np.asanyarray(saved.dataobj))) == expected_voxels
        pair_rows = []
        for rank, component_id in enumerate(sorted(range(1, region_count + 1), key=lambda i: (-int(sizes[i]), i)), 1):
            component_slices = region_slices[component_id - 1]
            offsets = np.array([bbox[axis][0] + component_slices[axis].start for axis in range(3)])
            local_positions = np.argwhere(labels[component_slices] == component_id)
            centroid_xyz = local_positions.mean(axis=0) + offsets
            centroid_ras = nib.affines.apply_affine(first_image.affine, centroid_xyz)
            global_bbox = [
                [bbox[axis][0] + component_slices[axis].start, bbox[axis][0] + component_slices[axis].stop - 1]
                for axis in range(3)
            ]
            pair_rows.append({
                "pair": pair,
                "rank_by_voxels": rank,
                "component_id_26": component_id,
                "voxels": int(sizes[component_id]),
                "volume_ml": int(sizes[component_id]) * voxel_volume_ml,
                "centroid_R_mm": float(centroid_ras[0]),
                "centroid_A_mm": float(centroid_ras[1]),
                "centroid_S_mm": float(centroid_ras[2]),
                "centroid_ijk_xyz_zero_based": json.dumps(centroid_xyz.tolist()),
                "bbox_xyz_zero_based_inclusive": json.dumps(global_bbox),
                "binary_overlap_mask": str(destination),
            })
        assert sum(row["voxels"] for row in pair_rows) == expected_voxels
        assert len(pair_rows) == expected_regions
        rows.extend(pair_rows)
        print(json.dumps({"pair": pair, "voxels": voxel_count, "regions_26": region_count, "output": str(destination)}, ensure_ascii=False), flush=True)
        del labels, cropped, first_image, second_image, saved
        gc.collect()
    csv_path = RESULTS / "overlap_regions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} region rows; wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
