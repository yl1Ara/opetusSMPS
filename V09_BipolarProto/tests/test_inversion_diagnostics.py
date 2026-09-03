import unittest

import numpy as np

from DMPS_inversion_gui.diagnostics import (
    integrate_number_distribution,
    range_overlap_metrics,
)
from inv_funcs.cpc_loss import cpc_loss1


class InversionDiagnosticTests(unittest.TestCase):
    def test_unknown_3771_efficiency_is_not_silently_replaced_with_3010_curve(self):
        np.testing.assert_allclose(
            cpc_loss1(np.array([5e-9, 10e-9, 20e-9]), 293.15, 101325, cpc_type="3771"),
            np.ones(3),
        )

    def test_ntot_integrates_stitched_distribution_once(self):
        sizes = np.array([10.0, 100.0, 1000.0])
        concentration = np.array([100.0, 100.0, 100.0])

        self.assertAlmostEqual(
            integrate_number_distribution(sizes, concentration),
            200.0,
        )

    def test_range_overlap_reports_relative_seam(self):
        metrics = range_overlap_metrics([
            np.array([100.0, 100.0, np.nan]),
            np.array([np.nan, 120.0, 120.0]),
        ])

        self.assertEqual(metrics["overlap_bin_count"], 1)
        self.assertAlmostEqual(metrics["median_relative_seam"], 20.0 / 110.0)

    def test_ntot_does_not_bridge_disconnected_scan_ranges(self):
        sizes = np.array([10.0, 100.0, 1000.0, 10000.0])
        concentration = np.full(4, 100.0)
        parts = [
            np.array([100.0, 100.0, np.nan, np.nan]),
            np.array([np.nan, np.nan, 100.0, 100.0]),
        ]

        self.assertAlmostEqual(
            integrate_number_distribution(sizes, concentration, part_columns=parts),
            200.0,
        )


if __name__ == "__main__":
    unittest.main()
