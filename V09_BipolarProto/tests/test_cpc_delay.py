import unittest

import numpy as np
import pandas as pd

from DMPS_inversion_gui.cpc_delay import assign_cpc_samples_to_setpoints


class CpcDelayReplayTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
