import unittest

import numpy as np
import pandas as pd

from DMPS_inversion_gui.diagnostics import (
    brownian_coagulation_kernel,
    brownian_coagulation_sink,
    build_effective_zratio_diagnostics,
    build_mcc_growth_cross_checks,
    build_formation_rate_diagnostics,
    build_growth_rate_diagnostics,
    build_quality_dashboard,
    build_temporal_mode_diagnostics,
    distribution_bin_coverage,
    distribution_moments,
    fit_lognormal_modes,
    growth_models_from_settings,
    integrate_number_distribution,
    range_overlap_metrics,
    select_lognormal_mode_fit,
    sulfuric_acid_condensation_sink,
    track_modal_components,
    weighted_log_diameter_quantile,
)
from inv_funcs.cpc_loss import cpc_loss1


class InversionDiagnosticTests(unittest.TestCase):
    @staticmethod
    def polarity_consistency_fixture(ratios=(1.44, 1.44), cpc_types=("3010", "3010")):
        sizes = np.geomspace(20.0, 70.0, 12)
        times = pd.date_range("2026-08-01", periods=2, freq="1h", tz="UTC")
        base_a = np.linspace(100.0, 500.0, len(sizes))
        base_b = np.linspace(700.0, 150.0, len(sizes))
        positive = np.column_stack([base_a * ratios[0], base_b * ratios[1]])
        negative = np.column_stack([base_b, base_a])
        return [{
            "kind": "heatmap", "method": "gunn woessner mod", "polarity": "positive",
            "x": times, "y": sizes, "Z": positive,
            "scan_id": ["scan-a", "scan-b"], "zratio_used": [1.0, 1.0],
            "cpc_type": list(cpc_types), "correction_mode": "Transport delay",
        }, {
            "kind": "heatmap", "method": "gunn woessner mod", "polarity": "negative",
            "x": times[::-1], "y": sizes, "Z": negative,
            "scan_id": ["scan-b", "scan-a"], "zratio_used": [1.0, 1.0],
            "cpc_type": list(cpc_types)[::-1], "correction_mode": "Transport delay",
        }]

    def test_effective_zratio_recovers_known_ratio_and_pairs_by_scan_id(self):
        result = self.polarity_consistency_fixture()
        original = np.asarray(result[0]["Z"]).copy()

        diagnostic = build_effective_zratio_diagnostics(result)

        self.assertEqual(diagnostic["status"], "ok")
        self.assertEqual(diagnostic["accepted_pair_count"], 2)
        self.assertAlmostEqual(diagnostic["median_effective_zratio"], 1.2, places=10)
        self.assertAlmostEqual(diagnostic["best_candidate_zratio"], 1.2, delta=0.02)
        self.assertEqual({row["scan_id"] for row in diagnostic["rows"]}, {"scan-a", "scan-b"})
        np.testing.assert_array_equal(result[0]["Z"], original)

    def test_effective_zratio_reports_pair_spread(self):
        diagnostic = build_effective_zratio_diagnostics(
            self.polarity_consistency_fixture(ratios=(1.0, 2.25))
        )

        estimates = sorted(row["effective_zratio"] for row in diagnostic["rows"])
        np.testing.assert_allclose(estimates, [1.0, 1.5])
        self.assertLess(diagnostic["effective_zratio_p10"], diagnostic["effective_zratio_p90"])

    def test_effective_zratio_flags_cpc_dependent_estimates(self):
        diagnostic = build_effective_zratio_diagnostics(
            self.polarity_consistency_fixture(
                ratios=(1.0, 2.25), cpc_types=("3010", "HY09"),
            )
        )

        self.assertTrue(diagnostic["cpc_dependence_warning"])
        self.assertEqual(
            {item["cpc_type"] for item in diagnostic["cpc_type_summaries"]},
            {"3010", "HY09"},
        )

    def test_effective_zratio_reports_unpaired_scan(self):
        result = self.polarity_consistency_fixture()
        result[1]["Z"] = result[1]["Z"][:, :1]
        for key in ("x", "scan_id", "zratio_used", "cpc_type"):
            result[1][key] = result[1][key][:1]

        diagnostic = build_effective_zratio_diagnostics(result)

        rejected = next(row for row in diagnostic["rows"] if row["scan_id"] == "scan-a")
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("negative-voltage scan is missing", rejected["reason"])

    def test_effective_zratio_rejects_insufficient_common_support(self):
        result = self.polarity_consistency_fixture()

        diagnostic = build_effective_zratio_diagnostics(
            result, minimum_size_nm=68.0, maximum_size_nm=70.0,
        )

        self.assertEqual(diagnostic["status"], "failed")
        self.assertTrue(all(row["status"] == "rejected" for row in diagnostic["rows"]))
        self.assertTrue(all("common measured bins" in row["reason"] for row in diagnostic["rows"]))

    def test_effective_zratio_marks_candidate_range_boundary(self):
        diagnostic = build_effective_zratio_diagnostics(
            self.polarity_consistency_fixture(ratios=(25.0, 25.0))
        )

        self.assertTrue(diagnostic["optimum_at_candidate_boundary"])
        self.assertAlmostEqual(diagnostic["best_candidate_zratio"], 3.0)

    def test_effective_zratio_fails_cleanly_for_misaligned_metadata(self):
        result = self.polarity_consistency_fixture()
        result[0]["zratio_used"] = [1.0]

        diagnostic = build_effective_zratio_diagnostics(result)

        self.assertEqual(diagnostic["status"], "failed")
        self.assertIn("column metadata is inconsistent", diagnostic["reason"])

    def test_effective_zratio_rejects_mismatched_cpc_type(self):
        result = self.polarity_consistency_fixture()
        result[1]["cpc_type"][1] = "HY09"

        diagnostic = build_effective_zratio_diagnostics(result)

        rejected = next(row for row in diagnostic["rows"] if row["scan_id"] == "scan-a")
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("CPC types differ", rejected["reason"])

    def test_log_diameter_quantile_is_exact_for_uniform_log_distribution(self):
        sizes = np.geomspace(10.0, 100.0, 6)

        self.assertAlmostEqual(
            weighted_log_diameter_quantile(sizes, np.ones(len(sizes)), 0.5),
            np.sqrt(1000.0),
        )

    def test_growth_model_settings_migrate_and_preserve_empty_selection(self):
        options = ["Center D50", "Ridge peak", "Appearance time"]
        defaults = ["Center D50", "Ridge peak"]

        self.assertEqual(growth_models_from_settings({"growth_models": []}, options, defaults), [])
        self.assertEqual(
            growth_models_from_settings({"growth_method": "peak size"}, options, defaults),
            ["Ridge peak"],
        )
        self.assertEqual(
            growth_models_from_settings({"growth_method": "weighted centroid"}, options, defaults),
            ["Center D50"],
        )
        self.assertEqual(
            growth_models_from_settings({"growth_models": "bad"}, options, defaults),
            defaults,
        )

    def test_banana_tracks_recover_known_growth_rate(self):
        sizes = np.geomspace(5.0, 40.0, 100)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=12, freq="30min")
        columns = []
        for index in range(len(times)):
            if index < 3:
                columns.append(np.full(len(sizes), 0.1))
                continue
            hours_since_event = 0.5 * (index - 3)
            center = 8.0 + 3.0 * hours_since_event
            columns.append(0.1 + 100.0 * np.exp(-0.5 * ((sizes - center) / 2.0) ** 2))
        result = [{
            "kind": "heatmap",
            "method": "test",
            "polarity": "positive",
            "x": times,
            "y": sizes,
            "Z": np.column_stack(columns),
        }]

        diagnostics = build_growth_rate_diagnostics(
            result,
            growth_min_size_nm=6.0,
            growth_max_size_nm=30.0,
            growth_threshold_fraction=0.2,
            growth_models=[
                "Lower edge D25", "Center D50", "Upper edge D75",
                "Ridge peak", "Appearance time",
            ],
            method_label=lambda value: value,
        )

        by_model = {row["model"]: row for row in diagnostics}
        self.assertTrue({
            "Lower edge D25", "Center D50", "Upper edge D75", "Ridge peak",
        }.issubset(by_model))
        self.assertAlmostEqual(by_model["Ridge peak"]["growth_rate"], 3.0, delta=0.35)
        self.assertAlmostEqual(by_model["Center D50"]["growth_rate"], 3.0, delta=0.6)
        self.assertAlmostEqual(by_model["Appearance time"]["growth_rate"], 3.0, delta=0.6)
        self.assertGreater(by_model["Appearance time"]["time"].min(), times[3])
        self.assertTrue(any(
            timestamp not in times for timestamp in by_model["Appearance time"]["time"]
        ))
        self.assertGreater(by_model["Ridge peak"]["r2"], 0.95)
        self.assertLess(
            np.nanmedian(by_model["Lower edge D25"]["dp"]),
            np.nanmedian(by_model["Center D50"]["dp"]),
        )
        self.assertLess(
            np.nanmedian(by_model["Center D50"]["dp"]),
            np.nanmedian(by_model["Upper edge D75"]["dp"]),
        )

    def test_flat_heatmap_does_not_claim_growth_event(self):
        result = [{
            "kind": "heatmap",
            "method": "test",
            "polarity": "positive",
            "x": pd.date_range("2026-08-01", periods=8, freq="1h"),
            "y": np.geomspace(5, 30, 20),
            "Z": np.ones((20, 8)),
        }]

        self.assertEqual(
            build_growth_rate_diagnostics(
                result,
                growth_min_size_nm=6,
                growth_max_size_nm=30,
                growth_threshold_fraction=0.3,
                growth_models=["Center D50", "Ridge peak"],
                method_label=lambda value: value,
            ),
            [],
        )

    def test_growth_events_are_split_across_large_time_gap(self):
        sizes = np.geomspace(5.0, 50.0, 120)
        early = pd.date_range("2026-08-01T00:00:00Z", periods=8, freq="1h")
        late = pd.date_range("2026-08-02T06:00:00Z", periods=5, freq="1h")
        times = early.append(late)
        columns = []
        for index in range(len(times)):
            if index < 3:
                columns.append(np.full(len(sizes), 0.1))
                continue
            event_index = index - 3
            center = 8.0 + 3.0 * event_index
            columns.append(0.1 + 100.0 * np.exp(-0.5 * ((sizes - center) / 2.0) ** 2))

        diagnostics = build_growth_rate_diagnostics(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": np.column_stack(columns),
            }],
            growth_min_size_nm=6,
            growth_max_size_nm=50,
            growth_threshold_fraction=0.2,
            growth_models=["Ridge peak"],
            method_label=lambda value: value,
            growth_max_gap_minutes=90,
            growth_min_event_scans=4,
        )

        self.assertEqual([row["event_number"] for row in diagnostics], [1, 2])
        for row in diagnostics:
            self.assertAlmostEqual(row["growth_rate"], 3.0, delta=0.5)

    def test_low_activity_scan_separates_distinct_events(self):
        sizes = np.geomspace(5.0, 35.0, 100)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=12, freq="1h")
        columns = []
        for index in range(len(times)):
            if index < 3 or index == 7:
                columns.append(np.full(len(sizes), 0.1))
            elif index < 7:
                center = 8.0 + 2.0 * (index - 3)
                columns.append(0.1 + 100 * np.exp(-0.5 * ((sizes - center) / 1.5) ** 2))
            else:
                center = 9.0 + 2.0 * (index - 8)
                columns.append(0.1 + 100 * np.exp(-0.5 * ((sizes - center) / 1.5) ** 2))

        diagnostics = build_growth_rate_diagnostics(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": np.column_stack(columns),
            }],
            growth_min_size_nm=6,
            growth_max_size_nm=30,
            growth_threshold_fraction=0.2,
            growth_models=["Ridge peak"],
            method_label=lambda value: value,
            growth_min_event_scans=4,
        )

        self.assertEqual([row["event_number"] for row in diagnostics], [1, 2])

    def test_ridge_tracker_does_not_jump_to_second_enhanced_mode(self):
        sizes = np.geomspace(5.0, 35.0, 140)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=11, freq="1h")
        columns = [np.full(len(sizes), 0.1) for _ in range(3)]
        for index in range(8):
            low_center = 8.0 + index
            low_mode = 40 * np.exp(-0.5 * ((sizes - low_center) / 1.0) ** 2)
            high_mode = (
                100 * np.exp(-0.5 * ((sizes - 24.0) / 1.2) ** 2)
                if index >= 4 else 0.0
            )
            columns.append(0.1 + low_mode + high_mode)

        diagnostics = build_growth_rate_diagnostics(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": np.column_stack(columns),
            }],
            growth_min_size_nm=6,
            growth_max_size_nm=30,
            growth_threshold_fraction=0.2,
            growth_models=["Ridge peak"],
            method_label=lambda value: value,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertLess(np.max(diagnostics[0]["dp"]), 17.0)
        self.assertAlmostEqual(diagnostics[0]["growth_rate"], 1.0, delta=0.3)

    def test_minimum_event_scans_applies_to_valid_track_points(self):
        sizes = np.geomspace(5.0, 30.0, 80)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=10, freq="1h")
        columns = [np.full(len(sizes), 0.1) for _ in range(3)]
        for index in range(7):
            center = 8.0 + 2.0 * index
            column = 0.1 + 100 * np.exp(-0.5 * ((sizes - center) / 1.5) ** 2)
            if index >= 3:
                column[:] = np.nan
                column[:10] = 0.1
            columns.append(column)

        diagnostics = build_growth_rate_diagnostics(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": np.column_stack(columns),
            }],
            growth_min_size_nm=6,
            growth_max_size_nm=30,
            growth_threshold_fraction=0.2,
            growth_models=["Ridge peak"],
            method_label=lambda value: value,
            growth_min_event_scans=6,
        )

        self.assertEqual(diagnostics, [])

    def test_ridge_survives_component_truncation_at_size_boundary(self):
        sizes = np.geomspace(5.0, 30.0, 120)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=10, freq="1h")
        columns = [np.full(len(sizes), 0.1) for _ in range(3)]
        for index in range(7):
            center = 8.0 + index
            columns.append(
                0.1 + 100 * np.exp(-0.5 * ((sizes - center) / 5.0) ** 2)
            )

        diagnostics = build_growth_rate_diagnostics(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": np.column_stack(columns),
            }],
            growth_min_size_nm=8,
            growth_max_size_nm=24,
            growth_threshold_fraction=0.2,
            growth_models=["Ridge peak", "Center D50"],
            method_label=lambda value: value,
        )

        ridge = [row for row in diagnostics if row["model"] == "Ridge peak"]
        self.assertEqual(len(ridge), 1)
        self.assertGreaterEqual(ridge[0]["n_points"], 4)
        self.assertFalse(any(row["model"] == "Center D50" for row in diagnostics))

    def test_weaker_event_is_not_hidden_by_stronger_event(self):
        sizes = np.geomspace(5.0, 35.0, 100)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=14, freq="1h")
        columns = [np.full(len(sizes), 0.1) for _ in range(3)]
        for amplitude in (100.0, 25.0):
            for index in range(4):
                center = 8.0 + 2.0 * index
                columns.append(
                    0.1 + amplitude * np.exp(-0.5 * ((sizes - center) / 1.5) ** 2)
                )
            if amplitude == 100.0:
                columns.extend([np.full(len(sizes), 0.1) for _ in range(3)])

        diagnostics = build_growth_rate_diagnostics(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": np.column_stack(columns),
            }],
            growth_min_size_nm=6,
            growth_max_size_nm=30,
            growth_threshold_fraction=0.35,
            growth_models=["Ridge peak"],
            method_label=lambda value: value,
            growth_min_event_scans=4,
        )

        self.assertEqual([row["event_number"] for row in diagnostics], [1, 2])

    def test_configured_maximum_growth_rate_rejects_fast_fit(self):
        sizes = np.geomspace(5.0, 20.0, 180)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=12, freq="1min")
        columns = [np.full(len(sizes), 0.1) for _ in range(3)]
        for index in range(9):
            center = 8.0 + 0.5 * index
            columns.append(
                0.1 + 100 * np.exp(-0.5 * ((sizes - center) / 0.4) ** 2)
            )

        diagnostics = build_growth_rate_diagnostics(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": np.column_stack(columns),
            }],
            growth_min_size_nm=6,
            growth_max_size_nm=18,
            growth_threshold_fraction=0.2,
            growth_models=["Ridge peak"],
            method_label=lambda value: value,
            growth_max_rate_nm_h=5.0,
        )

        self.assertEqual(diagnostics, [])

    def test_stationary_measurement_noise_does_not_create_growth(self):
        rng = np.random.default_rng(20260904)
        sizes = np.geomspace(5.0, 35.0, 80)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=24, freq="10min")
        for _ in range(50):
            noisy_flat = np.maximum(
                100.0 + rng.normal(0.0, 2.0, (len(sizes), len(times))), 0.01
            )
            diagnostics = build_growth_rate_diagnostics(
                [{
                    "kind": "heatmap", "method": "test", "polarity": "positive",
                    "x": times, "y": sizes, "Z": noisy_flat,
                }],
                growth_min_size_nm=6,
                growth_max_size_nm=30,
                growth_threshold_fraction=0.2,
                growth_models=["Center D50", "Ridge peak"],
                method_label=lambda value: value,
            )
            self.assertEqual(diagnostics, [])

    def test_short_cadence_track_below_rate_limit_is_retained(self):
        sizes = np.geomspace(5.0, 20.0, 220)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=65, freq="1min")
        columns = [np.full(len(sizes), 0.1) for _ in range(5)]
        for index in range(60):
            center = 8.0 + 3.0 * index / 60.0
            columns.append(
                0.1 + 100 * np.exp(-0.5 * ((sizes - center) / 0.35) ** 2)
            )

        diagnostics = build_growth_rate_diagnostics(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": np.column_stack(columns),
            }],
            growth_min_size_nm=6,
            growth_max_size_nm=15,
            growth_threshold_fraction=0.2,
            growth_models=["Ridge peak"],
            method_label=lambda value: value,
            growth_max_rate_nm_h=5.0,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertAlmostEqual(diagnostics[0]["growth_rate"], 3.0, delta=0.2)

    def test_background_subtraction_tracks_npf_mode_not_stationary_aitken_mode(self):
        sizes = np.geomspace(5.0, 40.0, 120)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=12, freq="30min")
        stationary = 300.0 * np.exp(-0.5 * ((sizes - 25.0) / 3.0) ** 2)
        columns = []
        for index in range(len(times)):
            event = 0.0
            if index >= 3:
                center = 8.0 + 2.5 * 0.5 * (index - 3)
                event = 100.0 * np.exp(-0.5 * ((sizes - center) / 1.5) ** 2)
            column = 0.1 + stationary + event
            column[index % len(sizes)] = np.nan
            columns.append(column)

        diagnostics = build_growth_rate_diagnostics(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": np.column_stack(columns),
            }],
            growth_min_size_nm=6,
            growth_max_size_nm=35,
            growth_threshold_fraction=0.2,
            growth_models=["Ridge peak", "Center D50"],
            method_label=lambda value: value,
        )

        by_model = {row["model"]: row for row in diagnostics}
        self.assertAlmostEqual(by_model["Ridge peak"]["growth_rate"], 2.5, delta=0.5)
        self.assertAlmostEqual(by_model["Center D50"]["growth_rate"], 2.5, delta=0.7)

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
            300.0,
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

    def test_stitched_ranges_do_not_assign_unmeasured_gap_to_end_bins(self):
        sizes = np.array([10.0, 20.0, 1000.0, 2000.0])
        concentration = np.full(4, 100.0)
        parts = [
            np.array([100.0, 100.0, np.nan, np.nan]),
            np.array([np.nan, np.nan, 100.0, 100.0]),
        ]

        self.assertAlmostEqual(
            integrate_number_distribution(sizes, concentration, part_columns=parts),
            200.0 * np.log10(2.0),
        )
        self.assertLess(
            distribution_bin_coverage(sizes, concentration, part_columns=parts),
            0.25,
        )

    def test_stitched_support_handles_internal_gap_after_missing_bins(self):
        sizes = np.array([5.0, 10.0, 20.0, 500.0, 1000.0, 2000.0])
        concentration = np.full(6, 100.0)
        parts = [np.array([np.nan, 100.0, 100.0, np.nan, 100.0, 100.0])]

        self.assertAlmostEqual(
            integrate_number_distribution(sizes, concentration, part_columns=parts),
            200.0 * np.log10(2.0),
        )

    def test_stitched_ranges_joining_at_one_center_keep_both_intervals(self):
        sizes = np.array([10.0, 20.0, 100.0])
        concentration = np.full(3, 100.0)
        parts = [
            np.array([100.0, 100.0, np.nan]),
            np.array([np.nan, 100.0, 100.0]),
        ]

        self.assertAlmostEqual(
            integrate_number_distribution(sizes, concentration, part_columns=parts),
            100.0,
        )
        self.assertAlmostEqual(
            distribution_bin_coverage(sizes, concentration, part_columns=parts),
            2.0 / 3.0,
        )

    def test_distribution_integration_reports_missing_bin_coverage(self):
        sizes = np.array([10.0, 20.0, 100.0, 1000.0])
        concentration = np.array([100.0, np.nan, 100.0, 100.0])
        widths = np.array([
            np.log10(20.0 / 10.0),
            0.5 * np.log10(100.0 / 10.0),
            0.5 * np.log10(1000.0 / 20.0),
            np.log10(1000.0 / 100.0),
        ])

        self.assertAlmostEqual(
            integrate_number_distribution(sizes, concentration),
            100.0 * np.sum(widths[[0, 2, 3]]),
        )
        self.assertAlmostEqual(
            distribution_bin_coverage(sizes, concentration),
            np.sum(widths[[0, 2, 3]]) / np.sum(widths),
        )

    def test_lognormal_mode_fit_recovers_two_modes_deterministically(self):
        sizes = np.geomspace(5.0, 300.0, 180)
        log_sizes = np.log10(sizes)
        concentration = (
            800 / (0.08 * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((log_sizes - np.log10(18.0)) / 0.08) ** 2)
            + 1200 / (0.11 * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((log_sizes - np.log10(95.0)) / 0.11) ** 2)
        )

        fitted = fit_lognormal_modes(sizes, concentration, 2)
        repeated = fit_lognormal_modes(sizes, concentration, 2)

        self.assertEqual(fitted["status"], "ok")
        np.testing.assert_allclose(
            [mode["mode_diameter_nm"] for mode in fitted["components"]],
            [18.0, 95.0],
            rtol=0.01,
        )
        self.assertGreater(fitted["r2"], 0.999)
        np.testing.assert_allclose(fitted["curve_total"], repeated["curve_total"])

    def test_automatic_modal_fit_uses_bic(self):
        sizes = np.geomspace(5.0, 100.0, 120)
        log_sizes = np.log10(sizes)
        concentration = 500 / (0.1 * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((log_sizes - np.log10(25.0)) / 0.1) ** 2
        )

        fitted = select_lognormal_mode_fit(sizes, concentration, 3)

        self.assertEqual(fitted["status"], "ok")
        self.assertEqual(fitted["number_of_modes"], 1)

    def test_automatic_modal_fit_does_not_add_poisson_noise_modes(self):
        rng = np.random.default_rng(20260904)
        sizes = np.geomspace(5.0, 150.0, 100)
        log_sizes = np.log10(sizes)
        expected = 1000 / (0.11 * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((log_sizes - np.log10(30.0)) / 0.11) ** 2
        )

        for _ in range(10):
            fitted = select_lognormal_mode_fit(
                sizes, rng.poisson(np.maximum(expected, 0)), 3
            )
            self.assertEqual(fitted["number_of_modes"], 1)
        for seed in (15, 72):
            fitted = select_lognormal_mode_fit(
                sizes,
                np.random.default_rng(seed).poisson(np.maximum(expected, 0)),
                3,
            )
            self.assertEqual(fitted["number_of_modes"], 1)

    def test_modal_tracking_preserves_identity_through_crossing(self):
        times = pd.date_range("2026-08-01", periods=6, freq="1h")
        fitted_scans = []
        for index, timestamp in enumerate(times):
            diameters = sorted((10.0 * 1.3 ** index, 40.0 / 1.3 ** index))
            components = [{
                "mode_diameter_nm": diameter,
                "area_cm3": 1000.0,
                "fit_range_area_cm3": 950.0,
                "geometric_std": 1.3,
            } for diameter in diameters]
            fitted_scans.append((timestamp, components, 0.99))

        rows = track_modal_components(fitted_scans)
        tracks = {}
        for row in rows:
            tracks.setdefault(row["track_id"], []).append(row["mode_diameter_nm"])

        self.assertEqual(len(tracks), 2)
        directions = sorted(np.sign(values[-1] - values[0]) for values in tracks.values())
        self.assertEqual(directions, [-1.0, 1.0])

    def test_all_failed_modal_scans_report_zero_success_fraction(self):
        times = pd.date_range("2026-08-01", periods=3, freq="1h")
        result = [{
            "kind": "heatmap", "method": "test", "polarity": "positive",
            "x": times, "y": np.geomspace(5.0, 100.0, 20),
            "Z": np.zeros((20, 3)),
        }]

        modes = build_temporal_mode_diagnostics(result)
        dashboard = build_quality_dashboard([{
            "scan_id": "scan", "nan_fraction": 0.0,
            "flow_rel_rmse": 0.0, "missing_polarity": "none",
        }], modes)

        self.assertEqual(modes[0]["failed_scan_count"], 3)
        self.assertEqual(modes[0]["successful_scan_count"], 0)
        self.assertEqual(dashboard[0]["modal_fit_success_fraction"], 0.0)

    def test_particle_moments_match_per_bin_fixture(self):
        sizes = np.array([10.0, 20.0, 40.0])
        width = np.log10(2.0)
        concentration = np.array([100.0, 200.0, 300.0]) / width

        moments = distribution_moments(sizes, concentration)

        self.assertAlmostEqual(moments["number_cm3"], 600.0)
        self.assertAlmostEqual(moments["number_mean_nm"], 17000.0 / 600.0)
        self.assertAlmostEqual(moments["geometric_mean_nm"], 25.1984209979)
        self.assertAlmostEqual(moments["surface_um2_cm3"], 1.79070781255)
        self.assertAlmostEqual(moments["volume_um3_cm3"], 0.0109432144100)

    def test_three_term_formation_budget_uses_upper_boundary_flux(self):
        sizes = np.array([3.0, 5.0, 8.0, 12.0, 20.0])
        times = pd.date_range("2026-08-01", periods=5, freq="1h", tz="UTC")
        z = np.column_stack([
            np.full(len(sizes), 100.0 + 10.0 * index)
            for index in range(len(times))
        ])
        result = [{
            "kind": "heatmap", "method": "test", "polarity": "positive",
            "x": times, "y": sizes, "Z": z,
        }]
        growth = [{
            "source_method": "test", "polarity": "positive",
            "model": "Center D50", "growth_rate": 2.0,
            "event_start": times[0], "event_end": times[-1],
        }]

        diagnostics = build_formation_rate_diagnostics(
            result,
            growth_min_size_nm=3.0,
            growth_max_size_nm=10.0,
            ntot_limit=1e6,
            method_label=lambda value: value,
            growth_diagnostics=growth,
        )

        self.assertEqual(len(diagnostics), 1)
        item = diagnostics[0]
        complete = item["three_term_budget_available"]
        self.assertTrue(np.all(complete))
        expected_outflux = (
            2.0 / 3600.0
            * (100.0 + 10.0 * np.arange(len(times)))
            / (10.0 * np.log(10.0))
        )
        np.testing.assert_allclose(item["growth_outflux_cm3_s1"], expected_outflux)
        np.testing.assert_allclose(
            item["formation_rate"][complete],
            item["accumulation_rate_cm3_s1"][complete]
            + item["growth_outflux_cm3_s1"][complete]
            + item["coagulation_loss_cm3_s1"][complete],
        )

        without_growth = build_formation_rate_diagnostics(
            result,
            growth_min_size_nm=3.0,
            growth_max_size_nm=10.0,
            ntot_limit=1e6,
            method_label=lambda value: value,
        )[0]
        self.assertTrue(np.all(np.isnan(without_growth["formation_rate"])))

    def test_formation_interval_can_use_interpolated_boundaries_with_one_center(self):
        sizes = np.array([2.0, 5.0, 20.0])
        times = pd.date_range("2026-08-01", periods=3, freq="1h", tz="UTC")
        result = [{
            "kind": "heatmap", "method": "test", "polarity": "positive",
            "x": times, "y": sizes, "Z": np.full((3, 3), 100.0),
        }]

        diagnostics = build_formation_rate_diagnostics(
            result, growth_min_size_nm=3.0, growth_max_size_nm=10.0,
            ntot_limit=1e6, method_label=lambda value: value,
        )

        self.assertEqual(len(diagnostics), 1)
        np.testing.assert_allclose(
            diagnostics[0]["concentration"], 100.0 * np.log10(10.0 / 3.0),
        )

    def test_properties_do_not_bridge_disconnected_measurement_ranges(self):
        sizes = np.array([10.0, 20.0, 1000.0, 2000.0])
        concentration = np.full(4, 100.0)
        parts = [
            np.array([100.0, 100.0, np.nan, np.nan]),
            np.array([np.nan, np.nan, 100.0, 100.0]),
        ]

        moments = distribution_moments(sizes, concentration, parts)
        cs = sulfuric_acid_condensation_sink(
            sizes, concentration, part_columns=parts
        )
        coags = brownian_coagulation_sink(
            sizes, concentration, 10.0, part_columns=parts
        )

        self.assertAlmostEqual(
            moments["number_cm3"], 200.0 * np.log10(2.0)
        )
        self.assertLess(moments["bin_coverage"], 0.25)
        self.assertLess(
            cs,
            sulfuric_acid_condensation_sink(sizes, concentration),
        )
        self.assertLess(
            coags,
            brownian_coagulation_sink(sizes, concentration, 10.0),
        )

    def test_condensation_and_coagulation_sinks_scale_with_concentration(self):
        sizes = np.array([10.0, 20.0, 40.0, 100.0])
        concentration = np.array([100.0, 200.0, 300.0, 400.0])
        cs = sulfuric_acid_condensation_sink(sizes, concentration)
        coags = brownian_coagulation_sink(sizes, concentration, 10.0)

        self.assertGreater(cs, 0)
        self.assertGreater(coags, 0)
        self.assertAlmostEqual(
            sulfuric_acid_condensation_sink(sizes, 2 * concentration), 2 * cs
        )
        self.assertAlmostEqual(
            brownian_coagulation_sink(sizes, 2 * concentration, 10.0), 2 * coags
        )
        self.assertAlmostEqual(
            brownian_coagulation_kernel(10.0, 100.0),
            brownian_coagulation_kernel(100.0, 10.0),
        )

    def test_mcc_cross_check_recovers_known_growth(self):
        sizes = np.geomspace(5.0, 35.0, 160)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=28, freq="15min")
        hours = np.arange(len(times)) * 0.25
        track_dp = 7.0 + 3.0 * hours
        z = np.column_stack([
            100 * np.exp(-0.5 * ((sizes - center) / 1.0) ** 2)
            for center in track_dp
        ])
        growth = [{
            "source_method": "test",
            "polarity": "positive",
            "event_id": "test:positive:event-1",
            "event_number": 1,
            "event_start": times[0],
            "event_end": times[-1],
            "dp": track_dp,
            "growth_rate": 3.0,
            "model": "Ridge peak",
            "background_quality": "adequate",
        }]

        checks = build_mcc_growth_cross_checks(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": z,
            }],
            growth,
            method_label=lambda value: value,
            maximum_growth_rate_nm_h=10.0,
            minimum_correlation=0.6,
        )

        self.assertEqual(len(checks), 1)
        self.assertAlmostEqual(checks[0]["growth_rate"], 3.0, delta=0.2)
        self.assertGreater(checks[0]["correlation"], 0.95)
        self.assertEqual(checks[0]["agreement"], "supportive")

    def test_mcc_does_not_claim_growth_for_stationary_mode(self):
        sizes = np.geomspace(5.0, 35.0, 120)
        times = pd.date_range("2026-08-01T00:00:00Z", periods=28, freq="15min")
        pulse = np.exp(-0.5 * ((np.arange(len(times)) - 14) / 3.0) ** 2)
        stationary = np.exp(-0.5 * ((sizes - 15.0) / 2.0) ** 2)
        z = 100 * np.outer(stationary, pulse)
        growth = [{
            "source_method": "test", "polarity": "positive",
            "event_id": "test:positive:event-1", "event_number": 1,
            "event_start": times[0], "event_end": times[-1],
            "dp": np.linspace(9.0, 21.0, len(times)), "growth_rate": 3.0,
            "model": "Ridge peak", "background_quality": "adequate",
        }]

        checks = build_mcc_growth_cross_checks(
            [{
                "kind": "heatmap", "method": "test", "polarity": "positive",
                "x": times, "y": sizes, "Z": z,
            }],
            growth,
            method_label=lambda value: value,
            maximum_growth_rate_nm_h=15.0,
            minimum_correlation=0.6,
        )

        self.assertEqual(checks, [])


if __name__ == "__main__":
    unittest.main()
