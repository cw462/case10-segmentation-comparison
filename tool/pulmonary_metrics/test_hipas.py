"""Synthetic tests of stated conventions, empty masks, ROI and physical spacing."""

import json
import unittest

import numpy as np

from hipas_metrics import evaluate_hipas


class HiPaSMetricsTests(unittest.TestCase):
    def evaluate(self, p, g, spacing=(1.0, 1.0, 1.0), roi=None):
        return evaluate_hipas(p, g, spacing, 1, 2, roi=roi)

    def test_perfect_two_classes(self):
        g = np.zeros((9, 9, 9), dtype=np.uint8)
        g[2:7, 2, 2] = 1
        g[2:7, 6, 6] = 2
        out = self.evaluate(g, g)
        for name in ("artery", "vein"):
            for metric in ("DSC", "SEN", "SPE"):
                self.assertEqual(out["metrics"][f"{name}_{metric}"], 1.0)
            self.assertEqual(out["metrics"][f"{name}_FPR"], 0.0)
            self.assertEqual(out["metrics"][f"{name}_HD95_mm"], 0.0)
            self.assertEqual(out["metrics"][f"{name}_SL_voxels_proxy_pred"], 5.0)
            self.assertEqual(out["metrics"][f"{name}_SL_voxels_proxy_ratio"], 1.0)
        self.assertEqual(out["metrics"]["MCS"], 0.0)
        self.assertEqual(set(out["metrics"]), set(out["status"]))

    def test_swapped_classes_match_printed_mcs_denominator(self):
        g = np.zeros((5, 5, 5), dtype=np.uint8)
        g[1:4, 1, 1] = 1
        g[1:4, 3, 3] = 2
        p = np.where(g == 1, 2, np.where(g == 2, 1, 0)).astype(np.uint8)
        out = self.evaluate(p, g)
        self.assertEqual(out["metrics"]["MCS"], 0.5)
        self.assertEqual(out["metrics"]["artery_DSC"], 0.0)

    def test_empty_masks_are_serializable_and_explicit(self):
        z = np.zeros((3, 3, 3), dtype=np.uint8)
        out = self.evaluate(z, z)
        self.assertIsNone(out["metrics"]["artery_DSC"])
        self.assertEqual(out["status"]["artery_DSC"], "undefined_both_empty")
        self.assertIsNone(out["metrics"]["artery_SEN"])
        self.assertIsNone(out["metrics"]["artery_HD95_mm"])
        self.assertIsNone(out["metrics"]["artery_BC_proxy_ratio"])
        self.assertIsNone(out["metrics"]["MCS"])
        json.dumps(out, allow_nan=False)

    def test_one_empty_class_and_no_reference_negative(self):
        g = np.ones((3, 3, 3), dtype=np.uint8)
        p = np.zeros_like(g)
        out = self.evaluate(p, g)
        self.assertEqual(out["metrics"]["artery_DSC"], 0.0)
        self.assertEqual(out["metrics"]["artery_SEN"], 0.0)
        self.assertIsNone(out["metrics"]["artery_SPE"])
        self.assertIsNone(out["metrics"]["artery_FPR"])
        self.assertEqual(out["status"]["artery_HD95_mm"], "undefined_one_empty")

    def test_fpr_denominator_is_one_vs_rest(self):
        g = np.zeros((3, 3, 3), dtype=np.uint8)
        g[1, 1, 1] = 1
        g[2, 2, 2] = 2
        p = g.copy()
        p[0, 0, 0] = 1
        out = self.evaluate(p, g)["metrics"]
        self.assertAlmostEqual(out["artery_FPR"], 1 / 26)
        self.assertAlmostEqual(out["artery_FPR"], 1 - out["artery_SPE"])

    def test_hd95_uses_spacing_in_array_axis_order(self):
        g = np.zeros((5, 5, 5), dtype=np.uint8)
        p = np.zeros_like(g)
        g[1, 2, 2] = 1
        p[2, 2, 2] = 1
        self.assertEqual(self.evaluate(p, g, (2, 3, 4))["metrics"]["artery_HD95_mm"], 2.0)

    def test_hd95_uses_directional_area_weighted_percentiles(self):
        g = np.zeros((5, 5, 5), dtype=np.uint8)
        g[1, 1, 1] = 1
        p = g.copy()
        p[1, 1, 3] = 1
        # Added distant component occupies half of the prediction surface.
        # Directional 95th-percentile maximum is 4 mm, not pooled-point 3.6.
        hd = self.evaluate(p, g, (1, 1, 2))["metrics"]["artery_HD95_mm"]
        self.assertAlmostEqual(hd, 4.0)

    def test_roi_does_not_cut_surfaces_or_skeletons(self):
        g = np.zeros((7, 7, 7), dtype=np.uint8)
        g[1:4, 1, 1] = 1
        p = g.copy()
        p[5, 5, 5] = 1
        roi = np.zeros_like(g, dtype=bool)
        roi[:4, :4, :4] = True
        full, restricted = self.evaluate(p, g), self.evaluate(p, g, roi=roi)
        self.assertEqual(restricted["metrics"]["artery_DSC"], 1.0)
        self.assertLess(full["metrics"]["artery_DSC"], 1.0)
        for key in ("artery_HD95_mm", "artery_SL_voxels_proxy_pred", "artery_BC_proxy_pred"):
            self.assertEqual(full["metrics"][key], restricted["metrics"][key])
        self.assertEqual(restricted["protocol"]["roi_scope"], "voxel_confusion_MCS_and_AUC_only")

    def test_native_grid_skeleton_count_does_not_become_mm(self):
        g = np.zeros((9, 9, 9), dtype=np.uint8)
        g[2:7, 4, 4] = 1
        a, b = self.evaluate(g, g, (1, 1, 1)), self.evaluate(g, g, (4, 2, 7))
        self.assertEqual(a["metrics"]["artery_SL_voxels_proxy_gt"], 5.0)
        self.assertEqual(a["metrics"]["artery_SL_voxels_proxy_gt"], b["metrics"]["artery_SL_voxels_proxy_gt"])
        self.assertEqual(a["metrics"]["artery_BC_proxy_gt"], 0.0)
        self.assertIsNone(a["metrics"]["artery_BC_proxy_ratio"])

    def test_invalid_inputs_fail_explicitly(self):
        z = np.zeros((3, 3, 3), dtype=np.uint8)
        for spacing in (None, (1, 1), (1, 0, 1), (1, float("nan"), 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                self.evaluate(z, z, spacing)
        with self.assertRaises(ValueError):
            self.evaluate(z.astype(float) + 0.1, z)
        with self.assertRaises(ValueError):
            self.evaluate(z.astype(int) - 1, z)
        with self.assertRaises(ValueError):
            self.evaluate(z, z, roi=np.zeros_like(z))
        with self.assertRaises(ValueError):
            self.evaluate(z, z[:2])
        with self.assertRaises(ValueError):
            evaluate_hipas(z, z, (1, 1, 1), 1, 1)

    def test_auc_perfect_reverse_and_tied_probabilities(self):
        g = np.zeros((3, 3, 3), dtype=np.uint8)
        g[1, 1, 1] = 1
        for score, expected in (
            (np.where(g == 1, 0.9, 0.1), 1.0),
            (np.where(g == 1, 0.1, 0.9), 0.0),
            (np.full(g.shape, 0.5), 0.5),
        ):
            with self.subTest(expected=expected):
                out = evaluate_hipas(g, g, (1, 1, 1), 1, 2, artery_probability=score)
                self.assertEqual(out["metrics"]["artery_AUC"], expected)
                self.assertIsNone(out["metrics"]["vein_AUC"])
                self.assertEqual(out["status"]["vein_AUC"], "not_computed_probability_not_supplied")

    def test_auc_missing_scores_or_single_class_are_undefined(self):
        g = np.ones((3, 3, 3), dtype=np.uint8)
        missing = self.evaluate(g, g)
        self.assertIsNone(missing["metrics"]["artery_AUC"])
        out = evaluate_hipas(g, g, (1, 1, 1), 1, 2,
                             artery_probability=np.full(g.shape, 0.5),
                             vein_probability=np.full(g.shape, 0.5))
        self.assertEqual(out["status"]["artery_AUC"], "undefined_no_reference_negative")
        self.assertEqual(out["status"]["vein_AUC"], "undefined_no_reference_positive")

    def test_auc_uses_roi_and_validates_probability_grid(self):
        g = np.zeros((3, 3, 3), dtype=np.uint8)
        g[0, 0, 0] = 1
        score = np.ones(g.shape, dtype=float)
        score[0, 0, 0], score[0, 0, 1] = 0.5, 0.1
        roi = np.zeros(g.shape, dtype=bool)
        roi[0, 0, :2] = True
        out = evaluate_hipas(g, g, (1, 1, 1), 1, 2, roi,
                             artery_probability=score)
        self.assertEqual(out["metrics"]["artery_AUC"], 1.0)
        for invalid in (score[:2], score + 0.5, np.full(g.shape, np.nan)):
            with self.subTest(invalid_shape=invalid.shape), self.assertRaises(ValueError):
                evaluate_hipas(g, g, (1, 1, 1), 1, 2, artery_probability=invalid)


if __name__ == "__main__":
    unittest.main()
