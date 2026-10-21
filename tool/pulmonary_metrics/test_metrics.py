"""Synthetic regression checks for the declared metric conventions.

Run from this directory: python -m unittest -v test_metrics.py
These checks validate this implementation, not equivalence to unpublished
VesMamba evaluation settings or annotation quality in a clinical dataset.
"""

import unittest

import numpy as np

from vesmamba_metrics import evaluate_binary


METRICS = ("Dice", "clDice", "HD95_mm", "NSD")


def thin_branch():
    """A one-voxel-wide Y tree, with no two-by-two voxel blocks."""
    mask = np.zeros((32, 32, 32), dtype=bool)
    mask[4:15, 16, 16] = True
    for z in range(15, 24):
        offset = z - 14
        mask[z, 16 + offset, 16] = True
        mask[z, 16 - offset, 16] = True
    return mask


class BinaryMetricsTests(unittest.TestCase):
    def evaluate(self, pred, gt, spacing=(1.0, 1.0, 1.0), tolerance=0.0):
        result = evaluate_binary(pred, gt, spacing, tolerance)
        self.assertTrue(set(METRICS).issubset(result))
        self.assertIsInstance(result["status"], dict)
        return result

    def test_identical_thin_branch_is_perfect(self):
        gt = thin_branch()
        result = self.evaluate(gt.copy(), gt, spacing=(0.6, 0.8, 1.7))
        for name in ("Dice", "clDice", "NSD"):
            self.assertAlmostEqual(result[name], 1.0, places=12)
        self.assertAlmostEqual(result["HD95_mm"], 0.0, places=12)

    def test_trunk_gap_loses_overlap_and_skeleton_coverage(self):
        gt = thin_branch()
        pred = gt.copy()
        # Keep the distal tree but interrupt its path to the proximal trunk.
        pred[8:12, 16, 16] = False
        result = self.evaluate(pred, gt)
        expected_dice = 2 * pred.sum() / (pred.sum() + gt.sum())
        self.assertAlmostEqual(result["Dice"], expected_dice, places=12)
        for name in ("Dice", "clDice", "NSD"):
            self.assertGreater(result[name], 0.0)
            self.assertLess(result[name], 1.0)
        self.assertGreater(result["HD95_mm"], 0.0)
        # This does not assert that clDice directly counts disconnected paths.

    def test_isolated_false_positive_is_not_silently_removed(self):
        gt = thin_branch()
        pred = gt.copy()
        pred[1, 1, 1] = True
        result = self.evaluate(pred, gt)
        expected_dice = 2 * gt.sum() / (pred.sum() + gt.sum())
        self.assertAlmostEqual(result["Dice"], expected_dice, places=12)
        for name in ("Dice", "clDice", "NSD"):
            self.assertGreater(result[name], 0.0)
            self.assertLess(result[name], 1.0)
        self.assertTrue(np.isfinite(result["HD95_mm"]))
        # HD95 may suppress a sufficiently small outlier; do not require it
        # to equal the maximum distance to this isolated false-positive voxel.

    def test_anisotropic_shift_uses_physical_spacing(self):
        gt = np.zeros((16, 16, 16), dtype=bool)
        pred = np.zeros_like(gt)
        gt[6, 8, 8] = True
        pred[8, 8, 8] = True
        iso = self.evaluate(pred, gt, spacing=(1.0, 1.0, 1.0), tolerance=2.0)
        aniso = self.evaluate(pred, gt, spacing=(3.0, 1.0, 1.0), tolerance=2.0)
        # With surface-distance's voxel-corner surfels, the two directed
        # 95th percentiles for this single-voxel translation are 2 spacings.
        self.assertAlmostEqual(iso["HD95_mm"], 2.0, places=12)
        self.assertAlmostEqual(aniso["HD95_mm"], 6.0, places=12)
        self.assertAlmostEqual(iso["NSD"], 1.0, places=12)
        self.assertAlmostEqual(aniso["NSD"], 0.0, places=12)
        self.assertEqual(iso["Dice"], aniso["Dice"])
        self.assertEqual(iso["clDice"], aniso["clDice"])

    def test_empty_masks_have_explicit_nonperfect_conventions(self):
        empty = np.zeros((8, 8, 8), dtype=bool)
        nonempty = empty.copy()
        nonempty[2:6, 4, 4] = True
        both = self.evaluate(empty, empty)
        for name in METRICS:
            self.assertIsNone(both[name])
        for pred, gt in ((empty, nonempty), (nonempty, empty)):
            with self.subTest(pred_empty=not pred.any()):
                result = self.evaluate(pred, gt)
                for name in ("Dice", "clDice", "NSD"):
                    self.assertEqual(result[name], 0.0)
                self.assertIsNone(result["HD95_mm"])

    def test_invalid_values_and_geometry_are_rejected(self):
        gt = thin_branch()
        for value in (np.nan, np.inf, -1.0, 0.5, 2.0):
            invalid = gt.astype(float)
            invalid[0, 0, 0] = value
            for role in ("prediction", "reference"):
                with self.subTest(value=value, role=role):
                    args = (invalid, gt) if role == "prediction" else (gt, invalid)
                    with self.assertRaises(ValueError):
                        evaluate_binary(*args, (1.0, 1.0, 1.0), 1.0)
        for spacing in ((0, 1, 1), (-1, 1, 1), (np.nan, 1, 1),
                        (1, np.inf, 1), (1, 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                evaluate_binary(gt, gt, spacing, 1.0)
        for tolerance in (-0.1, np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                evaluate_binary(gt, gt, (1, 1, 1), tolerance)
        with self.assertRaises(ValueError):
            evaluate_binary(gt[:, :, 0], gt[:, :, 0], (1, 1, 1), 1.0)
        with self.assertRaises(ValueError):
            evaluate_binary(gt[:-1], gt, (1, 1, 1), 1.0)


if __name__ == "__main__":
    unittest.main()
