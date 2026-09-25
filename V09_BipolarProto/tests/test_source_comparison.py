import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import panel as pn

import online_inversion_viewer as viewer
from DMPS_inversion_gui import diagnostics as diag
from DMPS_inversion_gui.source_comparison import (
    build_comparison_figure,
    nearest_number_ratio,
    number_on_common_support,
)


class IndependentSourceComparisonTests(unittest.TestCase):
    @staticmethod
    def heatmap(sizes, values):
        return {
            "kind": "heatmap", "method": "gunn woessner mod", "polarity": "positive",
            "x": ["2026-09-23 12:00", "2026-09-23 12:30"],
            "y": sizes,
            "Z": np.asarray(values, dtype=float),
        }

    def test_uses_common_measured_diameters_without_extrapolating(self):
        ufsmps = self.heatmap([10, 20, 40], [[500, 500], [100, 200], [100, 200]])
        smps = self.heatmap([20, 40, 80], [[100, 200], [100, 200], [500, 500]])
        sources = {"UFSMPS": [ufsmps], "SMPS": [smps]}

        figure, note = build_comparison_figure(sources, "gunn woessner mod", "positive", 900)

        self.assertIn("20 and 40 nm", note)
        self.assertEqual(len(figure.data), 5)  # two heatmaps, two N series, one ratio
        self.assertEqual(figure.layout.coloraxis.cmax, 900)
        self.assertEqual(figure.layout.yaxis.type, "log")
        self.assertEqual(figure.layout.yaxis2.type, "log")
        widths = diag.distribution_support_widths(np.array([10, 20, 40], dtype=float))
        self.assertAlmostEqual(figure.data[1].y[0], 100 * (widths[1] + widths[2]))
        self.assertAlmostEqual(figure.data[1].y[1], 200 * (widths[1] + widths[2]))
        self.assertEqual(len(number_on_common_support(ufsmps, 20, 40)), 2)
        self.assertEqual(figure.data[-1].name, "SMPS / UFSMPS")
        self.assertAlmostEqual(figure.data[-1].y[0], 1.0)

    def test_ratio_does_not_pair_scans_more_than_fifteen_minutes_apart(self):
        ratios = nearest_number_ratio(
            ["2026-09-23 12:01", "2026-09-23 12:40"], [200, 200],
            ["2026-09-23 12:00"], [100],
        )
        self.assertEqual(ratios[0], 2.0)
        self.assertTrue(np.isnan(ratios[1]))

    def test_no_overlap_does_not_claim_a_number_comparison(self):
        sources = {
            "UFSMPS": [self.heatmap([2, 3], [[10, 10], [20, 20]])],
            "SMPS": [self.heatmap([30, 40], [[10, 10], [20, 20]])],
        }
        figure, note = build_comparison_figure(sources, "gunn woessner mod", "positive")
        self.assertEqual(len(figure.data), 2)
        self.assertIn("overlapping measured diameters", note)

    def test_missing_polarity_is_not_mislabelled(self):
        figure, note = build_comparison_figure(
            {"UFSMPS": [self.heatmap([2, 3], [[10, 10], [20, 20]])]},
            "gunn woessner mod", "negative",
        )
        self.assertIsNone(figure)
        self.assertIn("Run an inversion", note)

    def test_instrument_tabs_keep_independent_results_for_comparison(self):
        trace_a = self.heatmap([10, 20, 40], [[500, 500], [100, 200], [100, 200]])
        trace_b = self.heatmap([20, 40, 80], [[100, 200], [100, 200], [500, 500]])
        modules = {
            "Bipolar Pi (CSC)": SimpleNamespace(
                shared_state={"lock": threading.Lock(), "latest_inversion": [trace_a]},
                start_app=lambda: pn.Column(),
            ),
            "SMEAR III SMPS (CSC)": SimpleNamespace(
                shared_state={"lock": threading.Lock(), "latest_inversion": [trace_b]},
                start_app=lambda: pn.Column(),
            ),
        }
        previous_started = pn.state.cache.get("online_inversion_viewer_global_started")
        try:
            with (
                patch.object(viewer.global_app, "start_app"),
                patch.object(viewer, "_global_live_tab", return_value=pn.Column()),
                patch.object(
                    viewer, "_load_session_app",
                    side_effect=lambda initial_scan_source: modules[initial_scan_source],
                ) as load_app,
            ):
                tabs = viewer.start_multi_app()
                self.assertEqual(load_app.call_count, 0)
                tabs.active = 1  # bipolar
                tabs.active = 4  # SMEAR III SMPS
                tabs.active = 5  # cross-instrument comparison
                self.assertEqual(load_app.call_count, 2)
                plot = tabs.objects[5].objects[-1]
                self.assertIn("Independent instrument inversions", plot.object.layout.title.text)
                self.assertEqual(len(plot.object.data), 5)
        finally:
            if previous_started is None:
                pn.state.cache.pop("online_inversion_viewer_global_started", None)
            else:
                pn.state.cache["online_inversion_viewer_global_started"] = previous_started


if __name__ == "__main__":
    unittest.main()
