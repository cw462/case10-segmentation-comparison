"""Read-only input audit of before/after vessel-component lineage (26 neighbors)."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STRUCTURES = ("pulmonary_artery", "pulmonary_vein")
CATEGORIES = (
    "retained_only_in_new_main",
    "all_deleted",
    "retained_only_in_one_new_nonmain",
    "retained_in_multiple_new_components",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_binary(path: Path):
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"Expected a 3D integer mask: {path}")
    if int(data.min()) != 0 or int(data.max()) != 1:
        raise ValueError(f"Expected a nonempty 0/1 mask: {path}")
    binary = data.astype(np.bool_, copy=True)
    del data
    return image, binary


def bounding_box(mask):
    extent = []
    for axis in range(3):
        indices = np.flatnonzero(np.any(mask, axis=tuple(a for a in range(3) if a != axis)))
        extent.append((int(indices[0]), int(indices[-1]) + 1))
    return extent


def audit(structure: str):
    before_path = ROOT / "inputs" / "before" / f"{structure}.nii.gz"
    after_path = ROOT / "inputs" / "after" / f"{structure}.nii.gz"
    before_image, before_full = read_binary(before_path)
    after_image, after_full = read_binary(after_path)
    if before_image.shape != after_image.shape or not np.allclose(
        before_image.affine, after_image.affine, rtol=0, atol=1e-6
    ):
        raise ValueError(f"Before/after grids differ: {structure}")
    first_bbox, second_bbox = bounding_box(before_full), bounding_box(after_full)
    bbox = [
        (min(a[0], b[0]), max(a[1], b[1]))
        for a, b in zip(first_bbox, second_bbox)
    ]
    slices = tuple(slice(low, high) for low, high in bbox)
    before = np.array(before_full[slices], dtype=np.bool_, order="C", copy=True)
    after = np.array(after_full[slices], dtype=np.bool_, order="C", copy=True)
    del before_full, after_full
    gc.collect()

    before_labels = np.empty(before.shape, dtype=np.int32, order="C")
    after_labels = np.empty(after.shape, dtype=np.int32, order="C")
    neighbor_structure = np.ones((3, 3, 3), dtype=np.uint8)
    before_count = int(ndimage.label(before, structure=neighbor_structure, output=before_labels))
    after_count = int(ndimage.label(after, structure=neighbor_structure, output=after_labels))

    # Foreground-only vectors avoid int64 copies of full-volume labels in bincount.
    old_ids = before_labels[before]
    new_ids_at_old = after_labels[before]
    old_sizes = np.bincount(old_ids, minlength=before_count + 1)
    new_sizes = np.bincount(after_labels[after], minlength=after_count + 1)
    old_main = int(old_sizes.argmax())
    new_main = int(new_sizes.argmax())
    old_nonmain = old_ids != old_main
    outside_total = int(np.count_nonzero(old_nonmain))
    entered_main = int(np.count_nonzero(old_nonmain & (new_ids_at_old == new_main)))
    still_nonmain = int(np.count_nonzero(old_nonmain & (new_ids_at_old != 0) & (new_ids_at_old != new_main)))
    deleted = int(np.count_nonzero(old_nonmain & (new_ids_at_old == 0)))
    assert entered_main + still_nonmain + deleted == outside_total

    # Sorted compact transition table: each pair is (old component, new component),
    # with new component 0 meaning an original foreground voxel was removed.
    base = after_count + 1
    pair_keys = old_ids.astype(np.int64) * base + new_ids_at_old
    unique_pairs, transition_counts = np.unique(pair_keys, return_counts=True)
    transitions = [[] for _ in range(before_count + 1)]
    for pair_key, count in zip(unique_pairs.tolist(), transition_counts.tolist()):
        old_id, new_id = divmod(pair_key, base)
        transitions[old_id].append((new_id, count))

    rows = []
    categories = dict.fromkeys(CATEGORIES, 0)
    for old_id in range(1, before_count + 1):
        mapping = dict(transitions[old_id])
        removed = int(mapping.get(0, 0))
        retained_map = {key: value for key, value in mapping.items() if key != 0}
        retained = int(sum(retained_map.values()))
        if retained == 0:
            category = "all_deleted"
        elif len(retained_map) > 1:
            category = "retained_in_multiple_new_components"
        elif new_main in retained_map:
            category = "retained_only_in_new_main"
        else:
            category = "retained_only_in_one_new_nonmain"
        if old_id != old_main:
            categories[category] += 1
        assert retained + removed == int(old_sizes[old_id])
        rows.append({
            "structure": structure,
            "before_component_id": old_id,
            "before_is_main": old_id == old_main,
            "before_voxels": int(old_sizes[old_id]),
            "retained_voxels": retained,
            "deleted_voxels": removed,
            "retained_fraction": retained / int(old_sizes[old_id]),
            "retained_in_after_main_voxels": int(retained_map.get(new_main, 0)),
            "retained_in_after_nonmain_voxels": retained - int(retained_map.get(new_main, 0)),
            "number_of_after_components_containing_retained_voxels": len(retained_map),
            "after_component_voxel_map": json.dumps(retained_map, sort_keys=True),
            "category": category,
        })
    assert sum(categories.values()) == before_count - 1
    nonmain_rows = [row for row in rows if not row["before_is_main"]]
    assert sum(row["retained_in_after_main_voxels"] for row in nonmain_rows) == entered_main
    assert sum(row["retained_in_after_nonmain_voxels"] for row in nonmain_rows) == still_nonmain
    assert sum(row["deleted_voxels"] for row in nonmain_rows) == deleted
    original_main_row = next(row for row in rows if row["before_is_main"])
    added = int(np.count_nonzero(after & ~before))
    deleted_all = int(np.count_nonzero(before & ~after))
    retained_counts_by_new_component = np.bincount(new_ids_at_old, minlength=after_count + 1)
    added_counts_by_new_component = new_sizes[1:] - retained_counts_by_new_component[1:]
    assert int(added_counts_by_new_component.sum()) == added
    purely_added_components = [
        {"after_component_id": new_id, "voxels": int(new_sizes[new_id]), "is_main": new_id == new_main}
        for new_id in range(1, after_count + 1)
        if retained_counts_by_new_component[new_id] == 0
    ]
    volume_per_voxel_mm3 = float(abs(np.linalg.det(before_image.affine[:3, :3])))
    result = {
        "before_path": str(before_path),
        "after_path": str(after_path),
        "before_sha256": sha256(before_path),
        "after_sha256": sha256(after_path),
        "shape_xyz": list(before_image.shape),
        "union_foreground_bbox_xyz_half_open": bbox,
        "cropped_shape_xyz": list(before.shape),
        "estimated_resident_mask_and_label_arrays_bytes": int(before.nbytes + after.nbytes + before_labels.nbytes + after_labels.nbytes),
        "voxel_volume_mm3": volume_per_voxel_mm3,
        "before_components": before_count,
        "after_components": after_count,
        "before_main_component_id": old_main,
        "after_main_component_id": new_main,
        "before_foreground_voxels": int(old_sizes.sum()),
        "after_foreground_voxels": int(new_sizes.sum()),
        "added_voxels": added,
        "deleted_voxels": deleted_all,
        "after_components_containing_only_added_voxels": purely_added_components,
        "after_components_containing_only_added_voxels_count": len(purely_added_components),
        "after_components_containing_any_retained_voxel_count": after_count - len(purely_added_components),
        "original_main_retention": original_main_row,
        "original_nonmain_voxels": {
            "total": outside_total,
            "retained_in_after_main": entered_main,
            "retained_in_after_nonmain": still_nonmain,
            "deleted": deleted,
            "accounting_check_passed": True,
        },
        "original_nonmain_components": {
            "total": before_count - 1,
            "categories": categories,
            "accounting_check_passed": True,
        },
        "category_original_voxel_totals": {
            category: sum(row["before_voxels"] for row in nonmain_rows if row["category"] == category)
            for category in CATEGORIES
        },
    }
    del before_labels, after_labels, before, after, old_ids, new_ids_at_old, pair_keys
    gc.collect()
    return result, rows


def main():
    RESULTS.mkdir(exist_ok=True)
    all_rows = []
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "connectivity": 26,
        "comparison": "Original before-component membership followed to after component IDs; no resampling or input changes.",
        "categories_partition_original_nonmain_components": {
            "retained_only_in_new_main": "At least one old voxel survives; all surviving voxels lie in the new largest component. Some original voxels may have been deleted.",
            "all_deleted": "No original voxel from this old component remains foreground after editing.",
            "retained_only_in_one_new_nonmain": "At least one old voxel survives; all surviving voxels lie in exactly one new component other than the largest. It may have merged with another old nonmain component.",
            "retained_in_multiple_new_components": "Surviving old voxels occupy at least two new components, possibly including the largest.",
        },
        "interpretation_limits": [
            "Connectivity and voxel retention are not anatomical accuracy or branch-detection measurements.",
            "A lower component count alone cannot distinguish adding connections from deleting fragments.",
            "Categories describe surviving original voxels; they do not label new voxels as anatomically correct bridges.",
            "Component IDs are local to this cropped-volume run and should not be matched to IDs in separate audits.",
        ],
        "structures": {},
    }
    for structure in STRUCTURES:
        result, rows = audit(structure)
        summary["structures"][structure] = result
        all_rows.extend(rows)
        print(json.dumps({"structure": structure, "original_nonmain_voxels": result["original_nonmain_voxels"], "original_nonmain_components": result["original_nonmain_components"]}), flush=True)
    summary["complete"] = True
    with (RESULTS / "lineage_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    with (RESULTS / "lineage_components.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print("Lineage accounting checks passed; summary and component CSV written.", flush=True)


if __name__ == "__main__":
    main()
