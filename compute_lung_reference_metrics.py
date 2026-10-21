"""Actually evaluate lung with the original package and user-designated final GT."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


WORK = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
TOOL = WORK / "tool" / "pulmonary_metrics"
sys.path.insert(0, str(TOOL))

import numpy as np
from io_utils import check_grids, load_volume
from vesmamba_metrics import evaluate_vesmamba


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    paths = {stage: WORK / "inputs" / stage / "lung.nii.gz" for stage in ("before", "after")}
    original_hashes = {stage: digest(path) for stage, path in paths.items()}
    loaded = {stage: load_volume(path) for stage, path in paths.items()}
    spacing = check_grids(loaded)
    prediction, ground_truth = loaded["before"]["data"], loaded["after"]["data"]
    before_voxels, after_voxels = int(np.count_nonzero(prediction)), int(np.count_nonzero(ground_truth))
    assert prediction.shape == (512, 512, 657)
    assert before_voxels > 0 and after_voxels > 0
    equal = bool(np.array_equal(prediction, ground_truth))
    assert equal, "Expected unchanged lung, but the two input masks differ."
    print("LUNG_REFERENCE_METRICS_START: evaluating original full-grid inputs; no identity short circuit", flush=True)
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(30):
            print(f"LUNG_REFERENCE_METRICS_RUNNING elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    evaluation_started = time.perf_counter()
    try:
        package_result = evaluate_vesmamba(prediction, ground_truth, spacing, 1.0)
    finally:
        finished.set()
        worker.join(timeout=1)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    for metric in ("binary.Dice", "binary.clDice", "binary.NSD"):
        assert package_result["status"][metric] == "ok", (metric, package_result["status"][metric])
        assert package_result["metrics"][metric] is not None, metric
    final_hashes = {stage: digest(path) for stage, path in paths.items()}
    assert original_hashes == final_hashes, "Input file hash changed during evaluation."
    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "structure": "lung",
        "reference_role": "user_designated_supplemented_ground_truth",
        "prediction": "before supplementation",
        "ground_truth": "after supplementation, designated by the user for this evaluation",
        "inputs": {stage: str(path) for stage, path in paths.items()},
        "input_sha256": original_hashes,
        "input_files_unchanged": True,
        "grid_shape": list(prediction.shape),
        "spacing_mm_array_axes": list(spacing),
        "evaluation_extent": "entire identical original grid; only the original package's shared-background bounding-box crop and zero padding are applied internally",
        "before_foreground_voxels": before_voxels,
        "after_foreground_voxels": after_voxels,
        "voxelwise_identical": equal,
        "NSD_tolerance_mm": 1.0,
        "computation": "evaluate_vesmamba(prediction, ground_truth, spacing, 1.0) was executed; no metric was filled from the equality check",
        "metric_implementation_path": str(TOOL / "vesmamba_metrics.py"),
        "metric_implementation_sha256": digest(TOOL / "vesmamba_metrics.py"),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-image", "nibabel", "surface-distance")},
        "metrics": package_result["metrics"],
        "package_result": package_result,
        "evaluation_elapsed_seconds": evaluation_elapsed,
        "elapsed_seconds": time.perf_counter() - started,
        "complete": True,
    }
    output = WORK / "results" / "lung_reference_metrics.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "evaluation_elapsed_seconds": evaluation_elapsed, "elapsed_seconds": report["elapsed_seconds"], "output": str(output)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
