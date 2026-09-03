import unittest

import numpy as np
import pandas as pd

from DMPS_inversion_gui.cpc_delay import (
    ResponseKernelGroup,
    assign_cpc_samples_to_setpoints,
    build_response_kernel,
    deduplicate_cpc_rows,
    response_kernel_ill_conditioned,
    response_kernel_rejection_reason,
    solve_response_kernel_nnls,
)


class CpcDelayReplayTests(unittest.TestCase):
    def test_response_kernel_replays_one_and_two_second_boxcars(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        for response_window in (1.0, 2.0):
            with self.subTest(response_window=response_window):
                rows = [
                    self._kernel_row(base, 0, 10.0, np.nan, 1, base, response_window),
                    self._kernel_row(
                        base + pd.Timedelta(seconds=2), 1, 20.0, 100.0, 2,
                        base + pd.Timedelta(seconds=2), response_window,
                    ),
                ]
                result = build_response_kernel(
                    pd.DataFrame(rows), 99.0, 99.0, 99.0,
                    group_columns=("polarity", "scan_range"),
                )
                group = result.groups[("positive", 1)]

                np.testing.assert_allclose(group.matrix, [[1.0, 0.0]])
                self.assertEqual(
                    result.diagnostics["metadata_provenance"]["response_window"],
                    "scan rows (cpc_response_window_sec)",
                )
                self.assertEqual(
                    result.diagnostics["metadata_provenance"]["transport_delay"],
                    "scan rows (cpc_transport_delay_sec)",
                )

    def test_six_second_3010_window_uses_full_commanded_setpoint_interval(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = [
            self._kernel_row(base + pd.Timedelta(seconds=11), 0, 10, 40, 1, base, 6),
            self._kernel_row(
                base + pd.Timedelta(seconds=13), 1, 20, np.nan, 2,
                base + pd.Timedelta(seconds=12), 6,
            ),
        ]
        rows[0].update(
            point_valid_from=base.isoformat(),
            point_valid_until=np.nan,
            cpc_transport_delay_sec=5.0,
        )
        rows[1].update(
            point_valid_from=(base + pd.Timedelta(seconds=12)).isoformat(),
            point_valid_until=(base + pd.Timedelta(seconds=14)).isoformat(),
        )

        result = build_response_kernel(pd.DataFrame(rows), 99, 99, 99)

        np.testing.assert_allclose(result.groups[("positive", 1)].matrix, [[1.0, 0.0]])

    def test_response_kernel_final_hold_covers_last_endpoint(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = [
            self._kernel_row(base, 0, 10.0, np.nan, 1, base, 0.0),
            self._kernel_row(
                base + pd.Timedelta(seconds=2), 1, 20.0, np.nan, 2,
                base + pd.Timedelta(seconds=2), 0.0,
            ),
            self._kernel_row(
                base + pd.Timedelta(seconds=4), 1, 20.0, 250.0, 3,
                base + pd.Timedelta(seconds=2), 0.0, phase="final_hold",
            ),
        ]

        result = build_response_kernel(pd.DataFrame(rows), 0, 0, 99)
        group = result.groups[("positive", 1)]

        np.testing.assert_allclose(group.matrix, [[0.0, 1.0]])
        self.assertTrue(group.diagnostics["endpoint_coverage"]["last"])
        self.assertEqual(result.diagnostics["metadata_provenance"]["dwell"], "scan rows (point_dwell_sec)")

    def test_response_kernel_uses_explicit_validity_windows_and_rejects_settling_gap(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        first = self._kernel_row(base + pd.Timedelta(seconds=1), 0, 10, 100, 1, base, 0)
        first.update(
            point_valid_from=base.isoformat(),
            point_valid_until=(base + pd.Timedelta(seconds=2)).isoformat(),
        )
        gap = self._kernel_row(
            base + pd.Timedelta(seconds=3), 1, 20, 999, 2,
            base + pd.Timedelta(seconds=4), 0,
        )
        gap.update(
            point_valid_from=(base + pd.Timedelta(seconds=4)).isoformat(),
            point_valid_until=(base + pd.Timedelta(seconds=6)).isoformat(),
        )

        result = build_response_kernel(pd.DataFrame([first, gap]), 0, 0, 2)

        group = result.groups[("positive", 1)]
        self.assertEqual(group.sample_values.tolist(), [100.0])
        self.assertEqual(result.diagnostics["outside_window_discards"], 1)
        self.assertEqual(
            result.diagnostics["metadata_provenance"]["setpoint_end"],
            "scan rows (point_valid_until)",
        )

    def test_response_kernel_deduplicates_positive_ids_globally(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        first = self._kernel_row(base + pd.Timedelta(seconds=1), 0, 10, 100, 7, base, 0)
        second = self._kernel_row(
            base + pd.Timedelta(seconds=3), 1, 20, 999, 7,
            base + pd.Timedelta(seconds=2), 0,
        )
        second.update(polarity="negative", scan_range=2)
        first["acquisition_session_id"] = "session-a"
        second["acquisition_session_id"] = "session-a"
        second["cpc_sample_time"] = first["cpc_sample_time"]

        result = build_response_kernel(pd.DataFrame([first, second]), 0, 0, 2)

        self.assertEqual(result.diagnostics["duplicate_cpc_ids_ignored"], 1)
        self.assertEqual(result.groups[("positive", 1)].sample_values.tolist(), [100.0])
        self.assertEqual(result.groups[("negative", 2)].sample_values.tolist(), [])

    def test_response_kernel_does_not_deduplicate_ids_from_different_sessions(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        first = self._kernel_row(base + pd.Timedelta(seconds=1), 0, 10, 100, 1, base, 0)
        second = self._kernel_row(
            base + pd.Timedelta(seconds=3), 1, 20, 200, 1,
            base + pd.Timedelta(seconds=2), 0,
        )
        first["acquisition_session_id"] = "session-a"
        second["acquisition_session_id"] = "session-b"

        result = build_response_kernel(pd.DataFrame([first, second]), 0, 0, 2)

        self.assertEqual(result.diagnostics["duplicate_cpc_ids_ignored"], 0)
        self.assertEqual(result.groups[("positive", 1)].sample_values.tolist(), [100.0, 200.0])

    def test_mixed_schema_legacy_rows_use_sample_time_for_dedup_scope(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        legacy = self._kernel_row(base + pd.Timedelta(seconds=1), 0, 10, 100, 1, base, 0)
        current = self._kernel_row(
            base + pd.Timedelta(seconds=3), 1, 20, 200, 1,
            base + pd.Timedelta(seconds=2), 0,
        )
        legacy["acquisition_session_id"] = np.nan
        current["acquisition_session_id"] = "session-b"

        result = build_response_kernel(pd.DataFrame([legacy, current]), 0, 0, 2)

        self.assertEqual(result.diagnostics["duplicate_cpc_ids_ignored"], 0)
        self.assertEqual(result.groups[("positive", 1)].sample_values.tolist(), [100.0, 200.0])

    def test_mixed_schema_kernel_rows_fall_back_to_legacy_point_start(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        legacy = self._kernel_row(base + pd.Timedelta(seconds=1), 0, 10, 100, 1, base, 0)
        legacy["cpc_transport_delay_sec"] = 0.75
        legacy["point_valid_from"] = np.nan
        current = self._kernel_row(
            base + pd.Timedelta(seconds=3), 1, 20, 200, 2,
            base + pd.Timedelta(seconds=2), 0,
        )
        current["point_valid_from"] = (base + pd.Timedelta(seconds=2)).isoformat()

        result = build_response_kernel(pd.DataFrame([legacy, current]), 0, 0, 2)

        self.assertEqual(result.groups[("positive", 1)].sample_values.tolist(), [100.0, 200.0])
        self.assertIn(
            "point_valid_from -> point_start_time",
            result.diagnostics["metadata_provenance"]["setpoint_start"],
        )

    def test_response_kernel_widget_override_is_explicit_in_provenance(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        row = self._kernel_row(base + pd.Timedelta(seconds=1), 0, 10, 100, 1, base, 2)

        result = build_response_kernel(
            pd.DataFrame([row]), 0, 0, 2, override_timing=True
        )

        provenance = result.diagnostics["metadata_provenance"]
        self.assertEqual(provenance["transport_delay"], "widget override (transport delay)")
        self.assertEqual(provenance["response_window"], "widget override (response window)")
        self.assertEqual(provenance["dwell"], "widget override (dwell)")

    def test_response_kernel_discards_support_crossing_group_boundary(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        first = self._kernel_row(base, 0, 10, np.nan, 1, base, 1)
        second = self._kernel_row(
            base + pd.Timedelta(seconds=2.5), 1, 20, 100, 2,
            base + pd.Timedelta(seconds=2), 1,
        )
        second.update(polarity="negative", scan_range=2)

        result = build_response_kernel(pd.DataFrame([first, second]), 0, 1, 2)

        self.assertEqual(result.diagnostics["mixed_boundary_discards"], 1)
        self.assertEqual(result.diagnostics["accepted_samples"], 0)

    def test_direct_response_kernel_nnls_recovers_synthetic_distribution(self):
        transfer = np.array([[1.0, 0.2], [0.1, 1.0]])
        kernel = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        expected = np.array([120.0, 35.0])
        measured = kernel @ transfer @ expected

        recovered, fitted, diagnostics = solve_response_kernel_nnls(kernel, transfer, measured)

        np.testing.assert_allclose(recovered, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fitted, measured, rtol=1e-12, atol=1e-12)
        self.assertEqual(diagnostics["rank"], 2)

    def test_smooth_nnls_stabilizes_rank_deficient_fast_scan(self):
        transfer = np.eye(3)
        kernel = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5],
        ])
        expected = np.array([100.0, 75.0, 50.0])
        measured = kernel @ expected

        recovered, fitted, diagnostics = solve_response_kernel_nnls(
            kernel, transfer, measured, smoothness=0.1
        )

        np.testing.assert_allclose(fitted, measured, rtol=2e-3)
        np.testing.assert_allclose(recovered, expected, rtol=2e-3)
        self.assertEqual(diagnostics["rank"], 2)
        self.assertEqual(diagnostics["augmented_rank"], 3)
        self.assertEqual(diagnostics["regularization_rows"], 1)

    def test_overlapping_rolling_averages_report_effective_sample_count(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = [
            self._kernel_row(base + pd.Timedelta(seconds=6 + offset), 0, 10, 40, offset + 1, base, 6)
            for offset in range(3)
        ]
        rows.append(self._kernel_row(
            base + pd.Timedelta(seconds=10), 1, 20, np.nan, 4,
            base + pd.Timedelta(seconds=10), 6,
        ))

        result = build_response_kernel(pd.DataFrame(rows), 0, 6, 10)
        diagnostics = result.groups[("positive", 1)].diagnostics

        self.assertEqual(diagnostics["sample_count"], 3)
        self.assertLess(diagnostics["effective_independent_sample_count"], 3)
        self.assertGreaterEqual(diagnostics["effective_independent_sample_count"], 1)

    def test_solver_whitens_correlated_rolling_average_samples(self):
        recovered, fitted, diagnostics = solve_response_kernel_nnls(
            np.eye(2),
            np.eye(2),
            np.array([100.0, 50.0]),
            correlation=np.array([[1.0, 0.8], [0.8, 1.0]]),
        )

        np.testing.assert_allclose(recovered, [100.0, 50.0])
        np.testing.assert_allclose(fitted, [100.0, 50.0])
        self.assertTrue(diagnostics["correlation_whitening"])

    def test_smooth_solver_keeps_correlation_whitening(self):
        kernel = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ])
        measured = np.array([100.0, 70.0, 45.0, 30.0])
        identity_solution, _, _ = solve_response_kernel_nnls(
            kernel, np.eye(3), measured, smoothness=0.5, correlation=np.eye(4)
        )
        correlated_solution, _, diagnostics = solve_response_kernel_nnls(
            kernel,
            np.eye(3),
            measured,
            smoothness=0.5,
            correlation=np.array([
                [1.0, 0.8, 0.0, 0.0],
                [0.8, 1.0, 0.4, 0.0],
                [0.0, 0.4, 1.0, 0.8],
                [0.0, 0.0, 0.8, 1.0],
            ]),
        )

        self.assertFalse(np.allclose(identity_solution, correlated_solution))
        self.assertTrue(diagnostics["correlation_whitening"])

    def test_observational_conditioning_drives_warning(self):
        _, _, diagnostics = solve_response_kernel_nnls(
            np.diag([1.0, 1e-10, 1.0]),
            np.eye(3),
            np.ones(3),
            smoothness=0.5,
        )

        self.assertGreater(diagnostics["condition_number"], 1e8)
        self.assertLess(diagnostics["augmented_condition_number"], 1e8)
        self.assertTrue(response_kernel_ill_conditioned(diagnostics))

    def test_deduplication_preserves_invalid_ids(self):
        rows = pd.DataFrame({
            "cpc_sample_id": [np.nan, np.nan, 0, 0, 7, 7],
            "cpc_sample_time": [1, 1, 2, 2, 3, 3],
            "cpc_count": [1, 2, 3, 4, 5, 6],
        })

        deduplicated, duplicate_count, _ = deduplicate_cpc_rows(rows)

        self.assertEqual(deduplicated["cpc_count"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(duplicate_count, 1)

    def test_rank_deficient_response_kernel_is_rejected(self):
        group = ResponseKernelGroup(
            sizes_nm=np.array([10.0, 20.0]),
            sample_values=np.array([100.0]),
            matrix=np.array([[0.5, 0.5]]),
            sample_ids=np.array([1.0]),
            diagnostics={"endpoint_coverage": {"first": True, "last": True}},
        )

        self.assertEqual(
            response_kernel_rejection_reason(group),
            "kernel rank 1 is below 2 bins",
        )

    def test_regularization_does_not_make_grossly_underdetermined_kernel_usable(self):
        matrix = np.zeros((2, 20))
        matrix[0, 0] = 1.0
        matrix[1, -1] = 1.0
        group = ResponseKernelGroup(
            sizes_nm=np.geomspace(10, 1000, 20),
            sample_values=np.array([100.0, 50.0]),
            matrix=matrix,
            sample_ids=np.array([1.0, 2.0]),
            diagnostics={
                "endpoint_coverage": {"first": True, "last": True},
                "per_bin_coverage": {str(index): 1.0 for index in range(20)},
            },
        )

        self.assertIn("kernel rank 2 is below 20 bins", response_kernel_rejection_reason(group))

    def test_delayed_smooth_replay_has_every_bin_and_no_jigsaw(self):
        sizes = np.array([10.0, 18.0, 32.0, 56.0, 100.0, 700.0])
        expected = np.array([100.0, 130.0, 160.0, 190.0, 220.0, 250.0])
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = []

        # A 12 s delayed CPC value is logged during the following 10 s DMA event.
        rows.append(self._row(base + pd.Timedelta(seconds=5), 0, sizes[0], 999.0, 1, base))
        for source_index, value in enumerate(expected[:-1]):
            current_index = source_index + 1
            sample_time = base + pd.Timedelta(seconds=10 * current_index + 7)
            rows.append(
                self._row(
                    sample_time,
                    current_index,
                    sizes[current_index],
                    value,
                    source_index + 2,
                    base + pd.Timedelta(seconds=10 * current_index),
                )
            )
        rows.append(
            self._row(
                base + pd.Timedelta(seconds=67),
                len(sizes) - 1,
                sizes[-1],
                expected[-1],
                99,
                base + pd.Timedelta(seconds=50),
                phase="final_hold",
            )
        )

        result = assign_cpc_samples_to_setpoints(pd.DataFrame(rows), delay_seconds=12)

        np.testing.assert_allclose(result.cpc_by_size.index, sizes)
        np.testing.assert_allclose(result.cpc_by_size.to_numpy(), expected)
        self.assertTrue(np.all(np.diff(result.cpc_by_size.to_numpy()) > 0))
        self.assertEqual(result.cpc_by_size.index[-1], 700.0)
        self.assertEqual(result.diagnostics["source_fallback_bins"], [])

    def test_duplicate_valid_sample_id_is_ignored(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = [
            self._row(base + pd.Timedelta(seconds=2), 0, 10.0, 100.0, 1, base),
            self._row(base + pd.Timedelta(seconds=12), 1, 20.0, 200.0, 2, base + pd.Timedelta(seconds=10)),
            self._row(base + pd.Timedelta(seconds=13), 1, 20.0, 9999.0, 2, base + pd.Timedelta(seconds=10)),
        ]

        result = assign_cpc_samples_to_setpoints(pd.DataFrame(rows), delay_seconds=0)

        self.assertEqual(result.cpc_by_size.loc[20.0], 200.0)
        self.assertEqual(result.diagnostics["duplicate_cpc_ids_ignored"], 1)

    def test_current_log_inference_and_first_scan_preserve_all_bins(self):
        sizes = [10.0, 20.0, 40.0, 700.0]
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = []
        for index, size in enumerate(sizes):
            point_start = base + pd.Timedelta(seconds=10 * index)
            row_time = point_start + pd.Timedelta(seconds=5)
            rows.append({
                "time": row_time,
                "cpc_sample_time": row_time.isoformat(),
                "cpc_sample_id": index + 1,
                "abs_size_nm": size,
                "cpc_float": 50.0 + 10.0 * index,
                "point_elapsed_sec": 2.0,
                "phase": "measuring" if index < len(sizes) - 1 else "final_hold",
            })

        result = assign_cpc_samples_to_setpoints(
            pd.DataFrame(rows), delay_seconds=5.0, settling_seconds=3.0
        )

        self.assertEqual(result.cpc_by_size.index.tolist(), sizes)
        self.assertFalse(result.cpc_by_size.isna().any())
        self.assertEqual(result.cpc_by_size.loc[700.0], 80.0)
        self.assertEqual(result.diagnostics["event_source"], "contiguous size runs")

    def test_transport_assignment_rejects_sample_in_explicit_settling_gap(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        first = self._row(base + pd.Timedelta(seconds=1), 0, 10, 100, 1, base)
        second = self._row(
            base + pd.Timedelta(seconds=3), 1, 20, 999, 2,
            base + pd.Timedelta(seconds=4),
        )
        first.update(
            point_valid_from=base.isoformat(),
            point_valid_until=(base + pd.Timedelta(seconds=2)).isoformat(),
        )
        second.update(
            point_valid_from=(base + pd.Timedelta(seconds=4)).isoformat(),
            point_valid_until=(base + pd.Timedelta(seconds=6)).isoformat(),
        )

        result = assign_cpc_samples_to_setpoints(pd.DataFrame([first, second]), delay_seconds=0)

        self.assertEqual(result.cpc_by_size.loc[10.0], 100.0)
        self.assertTrue(np.isnan(result.cpc_by_size.loc[20.0]))
        self.assertEqual(result.diagnostics["samples_outside_validity_windows"], 1)

    def test_transport_assignment_prefers_row_delay_unless_overridden(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = [
            self._row(base + pd.Timedelta(seconds=7), 0, 10, 100, 1, base),
            self._row(
                base + pd.Timedelta(seconds=12), 1, 20, 200, 2,
                base + pd.Timedelta(seconds=10),
            ),
        ]
        for row in rows:
            row["cpc_transport_delay_sec"] = 5.0

        metadata = assign_cpc_samples_to_setpoints(pd.DataFrame(rows), delay_seconds=0)
        overridden = assign_cpc_samples_to_setpoints(
            pd.DataFrame(rows), delay_seconds=0, override_timing=True
        )

        self.assertEqual(metadata.cpc_by_size.loc[10.0], 150.0)
        self.assertEqual(
            metadata.diagnostics["metadata_provenance"]["transport_delay"],
            "scan rows (cpc_transport_delay_sec)",
        )
        self.assertEqual(overridden.cpc_by_size.loc[10.0], 100.0)
        self.assertEqual(
            overridden.diagnostics["metadata_provenance"]["transport_delay"],
            "widget override (transport delay)",
        )

    def test_mixed_schema_transport_rows_fall_back_to_legacy_point_start(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        legacy = self._row(base + pd.Timedelta(seconds=1), 0, 10, 100, 1, base)
        legacy["point_valid_from"] = np.nan
        current = self._row(
            base + pd.Timedelta(seconds=3), 1, 20, 200, 2,
            base + pd.Timedelta(seconds=2),
        )
        current["point_valid_from"] = (base + pd.Timedelta(seconds=2)).isoformat()

        result = assign_cpc_samples_to_setpoints(
            pd.DataFrame([legacy, current]), delay_seconds=0.75
        )

        self.assertEqual(result.cpc_by_size.loc[10.0], 100.0)

    def test_missing_internal_bin_is_interpolated_but_endpoint_remains_missing(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = [
            self._row(base + pd.Timedelta(seconds=2), 0, 10.0, 100.0, 1, base),
            self._row(base + pd.Timedelta(seconds=12), 1, 20.0, np.nan, 2, base + pd.Timedelta(seconds=10)),
            self._row(base + pd.Timedelta(seconds=22), 2, 40.0, 300.0, 3, base + pd.Timedelta(seconds=20)),
            self._row(base + pd.Timedelta(seconds=32), 3, 700.0, np.nan, 4, base + pd.Timedelta(seconds=30)),
        ]

        result = assign_cpc_samples_to_setpoints(pd.DataFrame(rows), delay_seconds=0)

        self.assertTrue(np.isfinite(result.cpc_by_size.loc[20.0]))
        self.assertTrue(np.isnan(result.cpc_by_size.loc[700.0]))
        self.assertIn(20.0, result.diagnostics["interpolated_bins"])
        self.assertIn(700.0, result.diagnostics["edge_missing_bins"])

    def test_unassigned_endpoint_does_not_reuse_an_assigned_sample(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = [
            self._row(base + pd.Timedelta(seconds=2), 0, 10.0, 100.0, 1, base),
            self._row(base + pd.Timedelta(seconds=12), 1, 20.0, 200.0, 2, base + pd.Timedelta(seconds=10)),
            self._row(base + pd.Timedelta(seconds=22), 2, 40.0, 300.0, 3, base + pd.Timedelta(seconds=20)),
        ]

        result = assign_cpc_samples_to_setpoints(pd.DataFrame(rows), delay_seconds=5)

        self.assertTrue(np.isnan(result.cpc_by_size.loc[40.0]))
        self.assertEqual(result.diagnostics["source_fallback_bins"], [])
        self.assertIn(40.0, result.diagnostics["edge_missing_bins"])

    def test_assignment_crosses_range_and_polarity_boundaries_before_partitioning(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = []
        events = [
            (0, 10.0, "positive", 1, 999.0),
            (1, 20.0, "positive", 2, 100.0),
            (2, 30.0, "negative", 2, 200.0),
            (3, 40.0, "negative", 2, 300.0),
            (3, 40.0, "negative", 2, 400.0),
        ]
        sample_offsets = [2, 12, 22, 32, 42]
        for sample_id, ((point_index, size, polarity, scan_range, value), offset) in enumerate(
            zip(events, sample_offsets), start=1
        ):
            row = self._row(
                base + pd.Timedelta(seconds=offset),
                point_index,
                size,
                value,
                sample_id,
                base + pd.Timedelta(seconds=10 * point_index),
            )
            row.update(polarity=polarity, scan_range=scan_range)
            rows.append(row)

        result = assign_cpc_samples_to_setpoints(
            pd.DataFrame(rows),
            delay_seconds=6,
            group_columns=("polarity", "scan_range"),
        )

        self.assertEqual(result.cpc_by_size.loc[("positive", 1, 10.0)], 100.0)
        self.assertEqual(result.cpc_by_size.loc[("positive", 2, 20.0)], 200.0)
        self.assertEqual(result.cpc_by_size.loc[("negative", 2, 30.0)], 300.0)
        self.assertEqual(result.cpc_by_size.loc[("negative", 2, 40.0)], 400.0)
        self.assertEqual(result.diagnostics["source_fallback_bins"], [])

    def test_existing_schema_splits_equal_sizes_at_group_boundary(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = pd.DataFrame([
            {
                "time": base.isoformat(), "abs_size_nm": 40.0, "cpc_float": 100.0,
                "cpc_sample_id": 1, "polarity": "positive", "scan_range": 1,
            },
            {
                "time": (base + pd.Timedelta(seconds=10)).isoformat(),
                "abs_size_nm": 40.0, "cpc_float": 200.0, "cpc_sample_id": 2,
                "polarity": "negative", "scan_range": 2,
            },
        ])

        result = assign_cpc_samples_to_setpoints(
            rows, delay_seconds=0, group_columns=("polarity", "scan_range")
        )

        self.assertEqual(result.cpc_by_size.loc[("positive", 1, 40.0)], 100.0)
        self.assertEqual(result.cpc_by_size.loc[("negative", 2, 40.0)], 200.0)
        self.assertEqual(result.diagnostics["event_count"], 2)

    def test_empty_group_does_not_reuse_sample_assigned_to_previous_group(self):
        base = pd.Timestamp("2026-08-01T12:00:00Z")
        rows = []
        for point_index, size, scan_range, value, offset in [
            (0, 10.0, 1, 999.0, 2),
            (1, 20.0, 2, 100.0, 12),
        ]:
            row = self._row(
                base + pd.Timedelta(seconds=offset), point_index, size, value,
                point_index + 1, base + pd.Timedelta(seconds=10 * point_index),
            )
            row.update(polarity="positive", scan_range=scan_range)
            rows.append(row)

        result = assign_cpc_samples_to_setpoints(
            pd.DataFrame(rows), delay_seconds=6,
            group_columns=("polarity", "scan_range"),
        )

        self.assertEqual(result.cpc_by_size.loc[("positive", 1, 10.0)], 100.0)
        self.assertTrue(np.isnan(result.cpc_by_size.loc[("positive", 2, 20.0)]))

    @staticmethod
    def _row(sample_time, point_index, size, value, sample_id, point_start, phase="measuring"):
        return {
            "time": sample_time.isoformat(),
            "cpc_sample_time": sample_time.isoformat(),
            "cpc_sample_id": sample_id,
            "abs_size_nm": size,
            "cpc_float": value,
            "point_index": point_index,
            "point_start_time": point_start.isoformat(),
            "phase": phase,
        }

    @classmethod
    def _kernel_row(
        cls, sample_time, point_index, size, value, sample_id, point_start,
        response_window, phase="measuring",
    ):
        row = cls._row(sample_time, point_index, size, value, sample_id, point_start, phase)
        row.update({
            "polarity": "positive",
            "scan_range": 1,
            "cpc_transport_delay_sec": 0.0,
            "cpc_response_window_sec": response_window,
            "point_dwell_sec": 2.0,
        })
        return row


if __name__ == "__main__":
    unittest.main()
