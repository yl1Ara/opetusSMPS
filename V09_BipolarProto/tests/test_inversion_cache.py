import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from DMPS_inversion_gui.inversion_cache import InversionCache, cache_key, files_fingerprint


class InversionCacheTests(unittest.TestCase):
    def test_round_trip_preserves_result_and_diagnostics(self):
        frame = pd.DataFrame({
            "abs_size_nm": [10.0, 20.0],
            "N_GWalpha": [100.0, 50.0],
            "N_GWalpha_std": [4.0, 3.0],
        })
        frame.attrs["counting_covariance"] = np.array([[16.0, 1.0], [1.0, 9.0]])
        frame.attrs["assignment_diagnostics"] = {
            "fallback_bins": [10.0], "condition_number": float("inf"),
        }
        frame.attrs["sample_residual_rows"] = [{"residual_cpc": -1.5}]

        with tempfile.TemporaryDirectory() as directory:
            cache = InversionCache(directory)
            cache.store("a" * 64, frame)
            restored = cache.load("a" * 64)

            pd.testing.assert_frame_equal(restored, frame)
            np.testing.assert_allclose(
                restored.attrs["counting_covariance"],
                frame.attrs["counting_covariance"],
            )
            self.assertEqual(
                restored.attrs["assignment_diagnostics"]["fallback_bins"], [10.0],
            )
            self.assertTrue(np.isinf(
                restored.attrs["assignment_diagnostics"]["condition_number"],
            ))
            self.assertEqual(restored.attrs["sample_residual_rows"], frame.attrs["sample_residual_rows"])

    def test_key_changes_with_raw_data_or_settings(self):
        rows = pd.DataFrame({
            "time": pd.to_datetime(["2026-09-23T10:00:00"]),
            "size_nm": [10.0],
            "cpc_count": [100.0],
        })
        original = cache_key({"zratio": 1.1}, rows)

        changed_rows = rows.copy()
        changed_rows.loc[0, "cpc_count"] = 101.0
        self.assertNotEqual(original, cache_key({"zratio": 1.1}, changed_rows))
        self.assertNotEqual(original, cache_key({"zratio": 1.2}, rows))
        self.assertEqual(original, cache_key({"zratio": 1.1}, rows.copy()))

    def test_corrupt_artifact_is_removed_and_treated_as_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = InversionCache(directory)
            path = cache.path_for("b" * 64)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"not an npz file")

            self.assertIsNone(cache.load("b" * 64))
            self.assertFalse(path.exists())

    def test_code_fingerprint_changes_with_file_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.py"
            source.write_text("version = 1\n")
            original = files_fingerprint([source], root=root)

            source.write_text("version = 2\n")

            self.assertNotEqual(original, files_fingerprint([source], root=root))

    def test_clear_removes_only_cache_root(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            unrelated = parent / "raw_scan.csv"
            unrelated.write_text("scan data")
            cache = InversionCache(parent / "cache")
            frame = pd.DataFrame({"abs_size_nm": [10.0], "N_GWalpha": [1.0]})
            cache.store("c" * 64, frame)

            self.assertEqual(cache.stats()["entries"], 1)
            cache.clear()

            self.assertFalse(cache.root.exists())
            self.assertEqual(unrelated.read_text(), "scan data")


if __name__ == "__main__":
    unittest.main()
