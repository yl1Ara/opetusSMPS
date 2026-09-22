import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from DMPS_inversion_gui import online_app


class SmearIIIScanImportTests(unittest.TestCase):
    def test_dmps_scan_blocks_are_imported_with_dma_metadata(self):
        path = Path(__file__).resolve().parents[1] / "SMEARIII" / "DMPS007_20260629.scan"
        df = online_app.parse_smeariii_scan_file(path)

        self.assertFalse(df.empty)
        self.assertGreater(df["scan_id"].nunique(), 1)
        self.assertTrue((df["size_nm"] > 0).all())
        self.assertTrue(np.isclose(df["dma_length_m"].dropna().iloc[0], 0.28))
        self.assertTrue(np.isclose(df["dma_r1_m"].dropna().iloc[0], 0.025))
        self.assertTrue(np.isclose(df["dma_r2_m"].dropna().iloc[0], 0.033))
        self.assertTrue(np.isfinite(df["cpc_count"]).any())
        self.assertEqual(df["time"].iloc[0], pd.Timestamp("2026-06-29 01:02:26.669"))

    def test_ufsmps_duplicate_cpc_columns_use_last_counter_pair(self):
        path = Path(__file__).resolve().parents[1] / "SMEARIII" / "UFSMPS_proto_20260629.scan"
        df = online_app.parse_smeariii_scan_file(path)

        self.assertFalse(df.empty)
        self.assertTrue((df["size_nm"] > 0).all())
        self.assertTrue(np.isclose(df["dma_length_m"].dropna().iloc[0], 0.109))
        self.assertGreater(df["cpc_count"].max(), 0)

    def test_sum_times_are_aligned_to_rpi_summer_time(self):
        path = Path(__file__).resolve().parents[1] / "SMEARIII" / "DMPS007_20260629.sum"
        df = online_app.load_smeariii_sum_file(path)

        self.assertEqual(df["time"].iloc[0], pd.Timestamp("2026-06-29 01:02:13"))


if __name__ == "__main__":
    unittest.main()
