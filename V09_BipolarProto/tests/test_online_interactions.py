import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        selected_status = pn.pane.Markdown()
        selected_plot = pn.pane.Plotly()
        online_app.render_roi_analysis(
            analysis, plot_pane=selected_plot, status_pane=selected_status,
            store=False,
        )
        self.assertIn("Selected ROI", selected_status.object)
        self.assertIsNotNone(selected_plot.object)
        self.assertEqual(online_app.saved_roi_analyses, [])
        self.assertEqual(len(analysis["scan_fits"]), 2)
        self.assertIn("exact Plotly-selected cells retained", analysis["selection_semantics"])

        online_app.render_roi_analysis(analysis)
        self.assertEqual(len(online_app.saved_roi_analyses), 1)
        self.assertEqual(online_app.saved_roi_analyses[0]["roi_id"], "ROI-1")

    def test_browser_selectable_layer_maps_points_back_to_heatmap_cells(self):
        result, heatmap = self.heatmap_fixture()
        figure = make_subplots(rows=1, cols=1)
        figure.add_trace(heatmap.data[0], row=1, col=1)
        online_app.add_heatmap_selection_layer(figure, result[0], row=1)
        layer = figure.data[1]
        self.assertEqual(layer.type, "scattergl")
        self.assertEqual(layer.meta["kind"], "inversion_selection")
        points = []
        for point_index, (size_index, time_index, value) in enumerate(layer.customdata):
            if 35 <= size_index < 55:
                # Panel forwards scalar pointNumber and customdata for scatter
                # selection; it discards heatmap's array-valued pointNumber.
                points.append({
                    "curveNumber": 1,
                    "pointNumber": point_index,
                    "customdata": [size_index, time_index, value],
                })
        analysis = online_app.analyze_heatmap_roi(
            {"points": points}, figure, result, "1",
        )
        self.assertEqual(analysis["selected_cell_count"], 40)
        self.assertEqual(analysis["selected_scan_count"], 2)

        second_scan_index = next(
            index for index, data in enumerate(layer.customdata)
            if int(data[0]) == 75 and int(data[1]) == 1
        )
        click = online_app.analyze_heatmap_click(
            {"points": [{"curveNumber": 1, "pointNumber": second_scan_index}]},
            figure, result, "1", 6.5, 80.0,
        )
        self.assertEqual(click["status"], "ok")
        self.assertAlmostEqual(click["components"][0]["mode_diameter_nm"], 40.0, delta=0.5)

    def test_csc_selection_without_customdata_uses_cell_coordinates(self):
        result, _ = self.heatmap_fixture()
        times = pd.DatetimeIndex([
            "2026-08-01T00:00:00.123456Z", "2026-08-01T00:30:00.123456Z",
        ])
        result[0]["x"] = times
        figure = make_subplots(rows=1, cols=1)
        figure.add_heatmap(
            x=times, y=result[0]["y"], z=result[0]["Z"],
            meta={"kind": "inversion_heatmap", "method": "test", "polarity": "positive"},
            row=1, col=1,
        )
        online_app.add_heatmap_selection_layer(figure, result[0], row=1)
        layer = figure.data[1]
        # A real Panel event on CSC had customdata=None. Its Plotly trace held
        # a serialized mapping, so indexing it by pointNumber raised KeyError.
        serialized = SimpleNamespace(meta=layer.meta, customdata={"bdata": "encoded"})
        browser_figure = SimpleNamespace(data=[figure.data[0], serialized])
        points = [
            {
                "curveNumber": 1, "pointNumber": index, "customdata": None,
                "x": pd.Timestamp(layer.x[index]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-2],
                "y": float(layer.y[index]),
            }
            for index, (size_index, _, _) in enumerate(layer.customdata)
            if 35 <= size_index < 55
        ]
        analysis = online_app.analyze_heatmap_roi(
            {"points": points}, browser_figure, result, "1",
        )
        self.assertEqual(analysis["selected_cell_count"], 40)
        self.assertEqual(analysis["selected_scan_count"], 2)

    def test_smear_ntot_overlay_shifts_our_plot_but_not_inversion_heatmap(self):
        times = pd.date_range("2026-09-25 12:00", periods=2, freq="10min")
        shifted = times - pd.Timedelta(seconds=130)
        result = [{
            "kind": "heatmap", "method": "test", "polarity": "positive",
            "x": times, "y": [10.0, 20.0, 40.0],
            "Z": np.full((3, 2), 100.0),
        }, {
            "kind": "ntot", "method": "test", "polarity": "positive",
            "x": times, "y": [100.0, 200.0], "y_measured": [110.0, 210.0],
        }]
        smear_cpc = pd.DataFrame({
            "time": shifted, "SMEARIII_CPC": [95.0, 195.0],
        })
        with (
            patch.object(online_app, "smear_comparison_time_offset_sec", SimpleNamespace(value=-130.0)),
            patch.object(online_app, "load_smeariii_sum_range", return_value=pd.DataFrame()),
            patch.object(online_app, "load_smeariii_cpc_for_times", return_value=smear_cpc),
        ):
            figure = online_app.plot_inversion_result(result)

        heatmap = next(trace for trace in figure.data if trace.type == "heatmap")
        our_ntot = next(trace for trace in figure.data if trace.name == "test Ntot positive")
        smear_ntot = next(trace for trace in figure.data if trace.name == "SMEAR III CPC")
        self.assertEqual(pd.Timestamp(heatmap.x[0]), times[0])
        self.assertEqual(pd.Timestamp(our_ntot.x[0]), shifted[0])
        self.assertEqual(pd.Timestamp(smear_ntot.x[0]), shifted[0])
        self.assertEqual(result[1]["x"][0], times[0])

    def test_apply_comparison_shift_replots_without_reinverting(self):
        previous = online_app.latest_inversion
        previous_status = online_app.status.object
        result, _ = self.heatmap_fixture()
        online_app.latest_inversion = result
        try:
            with (
                patch.object(online_app, "smear_comparison_time_offset_sec", SimpleNamespace(value=-130.0)),
                patch.object(online_app, "plot_inversion_result", return_value=go.Figure()) as plot,
                patch.object(online_app, "plot_difference_diagnostics", return_value=go.Figure()),
                patch.object(online_app, "plot_smps_timing_diagnostics", return_value=go.Figure()),
                patch.object(online_app, "publish_shared_state") as publish,
                patch.object(online_app, "run_inversion_calculation") as invert,
            ):
                online_app.refresh_smear_comparison_plots()
                plot.assert_called_once_with(result, preserve_interactions=True)
                invert.assert_not_called()
                self.assertIn("-130", publish.call_args.kwargs["status_text"])
        finally:
            online_app.latest_inversion = previous
            online_app.status.object = previous_status

    def test_selected_growth_roi_reports_measured_d50_slope(self):
        sizes = np.geomspace(5.0, 30.0, 80)
        times = pd.date_range("2026-09-25", periods=8, freq="30min")
        z = np.column_stack([
            10 + 300 * np.exp(-0.5 * ((sizes - (8 + index)) / 1.2) ** 2)
            for index in range(len(times))
        ])
        result = [{
            "kind": "heatmap", "method": "test", "polarity": "positive",
            "x": times, "y": sizes, "Z": z,
        }]
        figure = go.Figure(go.Heatmap(
            x=times, y=sizes, z=z,
            meta={"kind": "inversion_heatmap", "method": "test", "polarity": "positive"},
        ))
        points = [
            {"curveNumber": 0, "pointNumber": [size_index, time_index]}
            for time_index in range(len(times))
            for size_index in range(len(sizes)) if 6 <= sizes[size_index] <= 21
        ]
        analysis = online_app.analyze_heatmap_roi({"points": points}, figure, result, "1")
        growth = analysis["selected_growth"]
        self.assertEqual(growth["status"], "ok")
        self.assertAlmostEqual(growth["growth_rate_nm_h"], 2.0, delta=0.35)

        status, plot, growth_plot = pn.pane.Markdown(), pn.pane.Plotly(), pn.pane.Plotly()
        online_app.render_roi_analysis(
            analysis, plot_pane=plot, status_pane=status,
            growth_plot_pane=growth_plot, store=False,
        )
        self.assertIn("apparent D50 slope", status.object)
        self.assertEqual(len(growth_plot.object.data), 2)

    def test_growth_signal_separates_stationary_raw_peak_from_enhancement(self):
        sizes = np.geomspace(5.0, 40.0, 100)
        stationary = 500 * np.exp(-0.5 * ((sizes - 25.0) / 2.0) ** 2)
        times = pd.date_range("2026-09-25", periods=10, freq="30min")
        z = np.column_stack([
            stationary + 10 + (
                200 * np.exp(-0.5 * ((sizes - (8 + index)) / 1.5) ** 2)
                if index >= 3 else 0
            ) for index in range(len(times))
        ])
        trace = {
            "kind": "heatmap", "method": "test", "polarity": "positive",
            "x": times, "y": sizes, "Z": z,
        }
        self.assertAlmostEqual(sizes[np.argmax(z[:, -1])], 25.0, delta=1.0)

        figure = online_app.plot_growth_signal([trace], [])
        enhancement = np.asarray(figure.data[0].z, dtype=float)
        signal_sizes = np.asarray(figure.data[0].y, dtype=float)
        self.assertLess(abs(signal_sizes[np.argmax(enhancement[:, -1])] - 17), 2.0)
        self.assertLess(enhancement[np.argmin(abs(signal_sizes - 25)), -1], 1.0)

        candidate_times = times[3:]
        centers = np.arange(11.0, 18.0)
        track = {
            "source_method": "test", "polarity": "positive", "model": "Center D50",
            "event_id": "test:positive:event-1", "event_number": 1,
            "time": candidate_times, "dp": centers, "fit": centers,
            "growth_rate": 2.0, "fit_quality": "acceptable",
            "component_support": [
                {"time": timestamp, "min_nm": center - 2, "max_nm": center + 2}
                for timestamp, center in zip(candidate_times, centers)
            ],
        }
        figure = online_app.plot_growth_signal([trace], [track])
        self.assertEqual([item.type for item in figure.data], [
            "heatmap", "heatmap", "scatter", "scatter",
        ])
        component = np.asarray(figure.data[1].z, dtype=float)
        self.assertTrue(np.isnan(component[np.argmin(abs(signal_sizes - 25))]).all())
        self.assertTrue(np.isfinite(component[np.argmin(abs(signal_sizes - 17)), -1]))

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

    def test_scan_dma_geometry_overrides_legacy_viewer_defaults(self):
        scan = pd.DataFrame({
            "dma_length_m": [0.11, 0.11],
            "dma_r1_m": [0.025, 0.025],
            "dma_r2_m": [0.033, 0.033],
        })

        dma = online_app.get_dma(scan)

        self.assertEqual((dma.L, dma.r1, dma.r2), (0.11, 0.025, 0.033))

    def test_inconsistent_scan_dma_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent dma_length_m"):
            online_app.get_dma(pd.DataFrame({"dma_length_m": [0.11, 0.28]}))

        filtered, diagnostics = online_app.filter_complete_scans(pd.DataFrame({
            "scan_id": ["bad-geometry", "bad-geometry"],
            "Ntot": [False, False],
            "dma_length_m": [0.11, 0.28],
        }))
        self.assertTrue(filtered.empty)
        self.assertIn("inconsistent dma_length_m", diagnostics[0]["reason"])

    def test_incomplete_scan_is_removed_without_dropping_complete_scan(self):
        rows = []
        for scan_id, point_indices in (("complete", range(3)), ("interrupted", range(2))):
            for point_index in point_indices:
                rows.append({
                    "scan_id": scan_id,
                    "Ntot": False,
                    "point_index": point_index,
                    "expected_scan_points": 3,
                    "scan_complete": scan_id == "complete",
                    "point_valid_until": "2026-01-01" if point_index == 2 else np.nan,
                })

        filtered, diagnostics = online_app.filter_complete_scans(pd.DataFrame(rows))

        self.assertEqual(set(filtered["scan_id"]), {"complete"})
        rejected = next(row for row in diagnostics if row["scan_id"] == "interrupted")
        self.assertFalse(rejected["accepted"])

    def test_legacy_scan_remains_usable_when_mixed_with_new_metadata(self):
        frame = pd.DataFrame({
            "scan_id": ["legacy", "legacy", "new", "new"],
            "Ntot": [False] * 4,
            "point_index": [np.nan, np.nan, 0, 1],
            "scan_complete": [np.nan, np.nan, True, True],
            "expected_scan_points": [np.nan, np.nan, 2, 2],
            "point_valid_until": [np.nan, np.nan, np.nan, "2026-01-01"],
            "_completion_metadata_present": [False, False, True, True],
        })

        filtered, diagnostics = online_app.filter_complete_scans(frame)

        self.assertEqual(set(filtered["scan_id"]), {"legacy", "new"})
        self.assertTrue(all(row["accepted"] for row in diagnostics))

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
