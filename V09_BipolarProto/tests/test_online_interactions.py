import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from DMPS_inversion_gui import online_app


class OnlineInteractionTests(unittest.TestCase):
    def setUp(self):
        self.previous_modal_analysis = online_app.latest_modal_analysis
        self.previous_rois = list(online_app.saved_roi_analyses)
        online_app.saved_roi_analyses.clear()

    def tearDown(self):
        online_app.latest_modal_analysis = self.previous_modal_analysis
        online_app.saved_roi_analyses[:] = self.previous_rois

    def heatmap_fixture(self):
        sizes = np.geomspace(5.0, 100.0, 100)
        log_sizes = np.log10(sizes)
        first = 100 / (0.1 * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((log_sizes - np.log10(15.0)) / 0.1) ** 2
        )
        second = 1000 / (0.1 * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((log_sizes - np.log10(40.0)) / 0.1) ** 2
        )
        times = pd.DatetimeIndex(["2026-08-01T00:00:00Z"] * 2)
        result = [{
            "kind": "heatmap", "method": "test", "polarity": "positive",
            "x": times, "y": sizes, "Z": np.column_stack([first, second]),
        }]
        figure = go.Figure(go.Heatmap(
            x=times,
            y=sizes,
            z=np.column_stack([first, second]),
            meta={"kind": "inversion_heatmap", "method": "test", "polarity": "positive"},
        ))
        return result, figure

    def test_heatmap_point_number_selects_duplicate_timestamp_column(self):
        result, figure = self.heatmap_fixture()
        analysis = online_app.analyze_heatmap_click(
            {"points": [{
                "curveNumber": 0,
                "pointNumber": [50, 1],
                "x": "2026-08-01T00:00:00Z",
                "y": 40.0,
            }]},
            figure,
            result,
            "1",
            6.5,
            80.0,
        )

        self.assertEqual(analysis["status"], "ok")
        self.assertAlmostEqual(
            analysis["components"][0]["mode_diameter_nm"], 40.0, delta=0.5
        )

    def test_non_heatmap_click_is_ignored(self):
        figure = go.Figure(go.Scatter(x=[1], y=[2]))
        self.assertIsNone(online_app.analyze_heatmap_click(
            {"points": [{"curveNumber": 0, "x": 1, "y": 2}]},
            figure,
            [],
            "1",
            6.5,
            80.0,
        ))

    def test_heatmap_roi_preserves_selected_cells_and_scans(self):
        result, figure = self.heatmap_fixture()
        points = [
            {"curveNumber": 0, "pointNumber": [size_index, time_index]}
            for time_index in (0, 1)
            for size_index in range(35, 55)
        ]
        analysis = online_app.analyze_heatmap_roi(
            {"points": points}, figure, result, "1",
        )

        self.assertEqual(analysis["selected_cell_count"], 40)
        self.assertEqual(analysis["selected_scan_count"], 2)
        self.assertEqual(len(analysis["scan_fits"]), 2)
        self.assertIn("exact Plotly-selected cells retained", analysis["selection_semantics"])

        online_app.render_roi_analysis(analysis)
        self.assertEqual(len(online_app.saved_roi_analyses), 1)
        self.assertEqual(online_app.saved_roi_analyses[0]["roi_id"], "ROI-1")

    def test_heatmap_roi_does_not_fit_across_unselected_size_gap(self):
        result, figure = self.heatmap_fixture()
        points = [
            {"curveNumber": 0, "pointNumber": [size_index, time_index]}
            for time_index in (0, 1)
            for size_index in (*range(20, 30), *range(70, 80))
        ]

        analysis = online_app.analyze_heatmap_roi(
            {"points": points}, figure, result, "1",
        )

        self.assertEqual(analysis["selected_cell_count"], 40)
        self.assertEqual(len(analysis["diameter_nm"]), 10)
        self.assertTrue(all(row["selected_cell_count"] == 20 for row in analysis["scan_fits"]))
        self.assertTrue(all(row["fit_cell_count"] == 10 for row in analysis["scan_fits"]))

    def test_heatmap_roi_does_not_fit_across_disconnected_measurement_ranges(self):
        result, figure = self.heatmap_fixture()
        left = np.full((100, 2), np.nan)
        right = np.full((100, 2), np.nan)
        left[:50] = np.asarray(result[0]["Z"][:50])
        right[50:] = np.asarray(result[0]["Z"][50:])
        result[0]["part_columns"] = [
            [left[:, time_index], right[:, time_index]] for time_index in (0, 1)
        ]
        points = [
            {"curveNumber": 0, "pointNumber": [size_index, time_index]}
            for time_index in (0, 1) for size_index in range(40, 60)
        ]

        analysis = online_app.analyze_heatmap_roi(
            {"points": points}, figure, result, "1",
        )

        self.assertEqual(len(analysis["diameter_nm"]), 10)
        self.assertTrue(all(row["fit_cell_count"] == 10 for row in analysis["scan_fits"]))

    def test_custom_modal_panes_do_not_change_save_cache(self):
        result, figure = self.heatmap_fixture()
        analysis = online_app.analyze_heatmap_click(
            {"points": [{
                "curveNumber": 0, "pointNumber": [50, 1],
                "x": "2026-08-01T00:00:00Z", "y": 40.0,
            }]},
            figure,
            result,
            "1",
            6.5,
            80.0,
        )
        sentinel = {"source": "personal-session"}
        online_app.latest_modal_analysis = sentinel

        online_app.render_modal_analysis(
            analysis, pn.pane.Plotly(), pn.pane.Markdown()
        )

        self.assertIs(online_app.latest_modal_analysis, sentinel)

    def test_json_conversion_replaces_nonfinite_values(self):
        converted = online_app._json_safe({
            "values": np.array([1.0, np.nan, np.inf]),
            "scalar": np.float64(-np.inf),
        })

        self.assertEqual(converted, {"values": [1.0, None, None], "scalar": None})
        json.dumps(converted, allow_nan=False)

    def test_contamination_check_uses_charge_opposite_voltage_sign(self):
        scan = pd.DataFrame({
            "size_nm": [20, -20, 40, -40, 80, -80],
            "cpc_count": [10, 10, 20, 20, 30, 30],
        })
        requested_charges = []

        def fake_fraction(charge, diameter, *args):
            requested_charges.append(charge)
            return np.array([0.1 if abs(charge) == 1 else 0.001])

        with (
            patch.object(online_app.inv, "min_mob", return_value=np.array([40e-9])),
            patch.object(online_app.inv, "gunn_woessner_modified", side_effect=fake_fraction),
        ):
            ratio, _ = online_app.estimate_ion_mobility_ratio_for_scan(scan)

        self.assertTrue(np.isfinite(ratio))
        self.assertEqual(requested_charges, [-1, -2, 1, 2])

    def test_timestamped_ambient_conditions_match_and_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ambient.csv"
            pd.DataFrame({
                "time": ["2026-08-01T00:00:00Z"],
                "temperature_c": [10.0],
                "pressure_hpa": [990.0],
            }).to_csv(path, index=False)
            matched = online_app.load_timestamped_ambient_conditions(
                ["2026-08-01T00:10:00Z", "2026-08-01T02:00:00Z"],
                str(path), 293.15, 101325.0, tolerance_minutes=30,
            )

        self.assertAlmostEqual(matched.loc[0, "temperature_k"], 283.15)
        self.assertAlmostEqual(matched.loc[0, "pressure_pa"], 99000.0)
        self.assertIn("timestamped ambient CSV", matched.loc[0, "condition_source"])
        self.assertEqual(matched.loc[1, "condition_source"], "configured fallback")
        self.assertAlmostEqual(matched.loc[1, "temperature_k"], 293.15)

    def test_ambient_fallback_always_includes_unmatched_timestamp(self):
        times = ["2026-08-01T00:10:00Z"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.csv"
            empty_observations = Path(temporary_directory) / "ambient.csv"
            pd.DataFrame({
                "time": ["not-a-time"],
                "temperature_k": [293.15],
                "pressure_pa": [101325.0],
            }).to_csv(empty_observations, index=False)

            for path in ("", str(missing), str(empty_observations)):
                matched = online_app.load_timestamped_ambient_conditions(
                    times, path, 293.15, 101325.0,
                )
                self.assertIn("ambient_time", matched.columns)
                self.assertTrue(pd.isna(matched.loc[0, "ambient_time"]))
                self.assertEqual(matched.loc[0, "condition_source"], "configured fallback")

    def test_ambient_matching_preserves_invalid_and_mixed_timezone_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ambient.csv"
            pd.DataFrame({
                "time": ["2026-08-01 03:00:00", "2026-08-01T01:00:00Z"],
                "temperature_k": [280.0, 281.0],
                "pressure_pa": [99000.0, 99100.0],
            }).to_csv(path, index=False)
            matched = online_app.load_timestamped_ambient_conditions(
                ["2026-08-01T00:05:00Z", "not-a-time", "2026-08-01 04:05:00"],
                str(path), 293.15, 101325.0, tolerance_minutes=10,
            )

        self.assertEqual(len(matched), 3)
        self.assertAlmostEqual(matched.loc[0, "temperature_k"], 280.0)
        self.assertEqual(matched.loc[1, "condition_source"], "configured fallback")
        self.assertTrue(pd.isna(matched.loc[1, "time"]))
        self.assertAlmostEqual(matched.loc[2, "temperature_k"], 281.0)


if __name__ == "__main__":
    unittest.main()
