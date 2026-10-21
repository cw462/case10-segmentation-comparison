"""Meaningful synthetic checks; no patient data and no author-code equivalence claim."""
import json
import unittest

import numpy as np

from airrc_metrics import evaluate_airrc


class AirRCMetricsTests(unittest.TestCase):
    def test_multiclass_counts_and_class_swap(self):
        gt = np.array([1, 1, 2, 2, 0, 0, 0, 0]).reshape(2, 2, 2)
        pred = np.array([1, 2, 2, 0, 1, 0, 0, 0]).reshape(gt.shape)
        r = evaluate_airrc(pred, gt, (1, 1, 1), {"airway": 1, "artery": 2})
        self.assertEqual(r["metrics"]["airway.DSC"], 0.5)
        self.assertEqual(r["metrics"]["airway.Precision"], 0.5)
        self.assertEqual(r["metrics"]["airway.Recall"], 0.5)
        self.assertAlmostEqual(r["metrics"]["airway.FPR"], 1 / 6)
        self.assertEqual(r["metrics"]["artery.DSC"], 0.5)
        json.dumps(r, allow_nan=False)

    def test_empty_class_handling(self):
        z = np.zeros((3, 3, 3), dtype=np.uint8)
        r = evaluate_airrc(z, z, (1, 1, 1), {"airway": 1}, airway_label=1)
        self.assertIsNone(r["metrics"]["airway.DSC"])
        self.assertIsNone(r["metrics"]["airway.Precision"])
        self.assertIsNone(r["metrics"]["airway.Recall"])
        self.assertEqual(r["metrics"]["airway.FPR"], 0)
        self.assertIsNone(r["metrics"]["airway.BDR_gt80_independent"])
        gt = z.copy(); gt[1, 1, 1] = 1
        r = evaluate_airrc(z, gt, (1, 1, 1), {"airway": 1})
        self.assertEqual(r["metrics"]["airway.DSC"], 0)
        self.assertEqual(r["metrics"]["airway.Recall"], 0)
        self.assertIsNone(r["metrics"]["airway.Precision"])

    def test_missing_branch_labels_not_faked(self):
        gt = np.ones((2, 2, 2), dtype=int)
        r = evaluate_airrc(gt, gt, (1, 1, 1), {"airway": 1}, airway_label="airway")
        self.assertEqual(r["metrics"]["airway.DSC"], 1)
        self.assertIsNone(r["metrics"]["airway.BDR_gt80_independent"])
        self.assertIn("requires_fixed_reference_branch", r["status"]["airway.BDR_gt80_independent"])

    def test_strict_80_percent_and_matched_whole_length(self):
        gt = np.zeros((16, 8, 3), dtype=int)
        branches = gt.copy()
        branches[1:12, 1, 1] = 10  # 10 edges, spacing[0]=2 -> 20 mm
        branches[1:6, 5, 1] = 20   # 4 edges -> 8 mm
        gt[branches > 0] = 1
        pred = np.zeros_like(gt)
        pred[1:10, 1, 1] = 1       # 8/10 edges, exactly 80%, NOT detected
        r = evaluate_airrc(pred, gt, (2, 1, 1), {"airway": 1}, 1, branches)
        self.assertEqual(r["metrics"]["airway.BDR_gt80_independent"], 0)
        self.assertEqual(r["metrics"]["airway.DLR_matched_gt_length_independent"], 0)
        pred[10, 1, 1] = 1         # 9/10 edges, detected
        r = evaluate_airrc(pred, gt, (2, 1, 1), {"airway": 1}, 1, branches)
        self.assertEqual(r["metrics"]["airway.BDR_gt80_independent"], 0.5)
        self.assertAlmostEqual(r["metrics"]["airway.DLR_matched_gt_length_independent"], 20 / 28)
        self.assertNotAlmostEqual(r["metrics"]["airway.DLR_matched_gt_length_independent"], 18 / 28)
        self.assertEqual(r["protocol"]["branch_protocol"]["total_reference_length_mm"], 28)
        json.dumps(r, allow_nan=False)

    def test_anisotropic_diagonal_physical_length(self):
        gt = np.zeros((6, 6, 6), dtype=int)
        b = np.zeros_like(gt)
        for i in range(1, 5):
            gt[i, i, i] = 1; b[i, i, i] = 7
        r = evaluate_airrc(gt, gt, (2, 3, 4), {"airway": 1}, 1, b)
        self.assertAlmostEqual(r["protocol"]["branch_protocol"]["total_reference_length_mm"], 3 * np.sqrt(29))
        self.assertEqual(r["metrics"]["airway.BDR_gt80_independent"], 1)
        self.assertEqual(r["metrics"]["airway.DLR_matched_gt_length_independent"], 1)

    def test_invalid_branch_maps_rejected(self):
        gt = np.ones((5, 5, 5), dtype=int)
        single = np.zeros_like(gt); single[1, 1, 1] = 1
        with self.assertRaisesRegex(ValueError, "at least two"):
            evaluate_airrc(gt, gt, (1, 1, 1), {"airway": 1}, 1, single)
        corner = np.zeros_like(gt)
        corner[1, 1, 1] = corner[2, 1, 1] = corner[2, 2, 1] = 1
        with self.assertRaisesRegex(ValueError, "simple 26-neighbour path"):
            evaluate_airrc(gt, gt, (1, 1, 1), {"airway": 1}, 1, corner)
        bad = np.zeros_like(gt); bad[1, 1, 1] = 1; bad[3, 3, 3] = 1
        with self.assertRaisesRegex(ValueError, "simple 26-neighbour path"):
            evaluate_airrc(gt, gt, (1, 1, 1), {"airway": 1}, 1, bad)
        bad = np.zeros_like(gt); bad[1:3, 1:3, 1:3] = 1
        with self.assertRaises(ValueError):
            evaluate_airrc(gt, gt, (1, 1, 1), {"airway": 1}, 1, bad)
        gt[bad > 0] = 0
        with self.assertRaisesRegex(ValueError, "inside GT"):
            evaluate_airrc(gt, gt, (1, 1, 1), {"airway": 1}, 1, bad)

    def test_input_validation(self):
        z = np.zeros((2, 2, 2), dtype=int)
        for spacing in [(1, 0, 1), (1, np.nan, 1), (1, 1)]:
            with self.assertRaises(ValueError):
                evaluate_airrc(z, z, spacing, {"airway": 1})
        with self.assertRaises(ValueError):
            evaluate_airrc(z + 0.5, z, (1, 1, 1), {"airway": 1})
        with self.assertRaises(ValueError):
            evaluate_airrc(z, z, (1, 1, 1), {"airway": 1, "wall": 1})
        with self.assertRaises(ValueError):
            evaluate_airrc(z, z, (1, 1, 1), {"background": 0})


if __name__ == "__main__":
    unittest.main()
