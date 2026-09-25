import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from DMPS_inversion_gui import online_app


class SmearIIIScanImportTests(unittest.TestCase):
    def test_smps_gw_inversion_renders_with_microsecond_cpc_times(self):
        root = Path(__file__).resolve().parents[1] / "SMEARIII"
        scan = online_app.parse_smeariii_scan_file(root / "DMPS007_20260629.scan")
        one_scan = scan[scan["scan_id"] == scan["scan_id"].iloc[0]].copy()

        def fake_smear_cpc(times):
            matched_times = pd.to_datetime(times).astype("datetime64[us]")
            return pd.DataFrame({
                "time": matched_times,
                "SMEARIII_CPC": np.full(len(matched_times), 100.0),
            })

        with (
            patch.object(online_app, "scan_inversion_type", SimpleNamespace(value="SMPS")),
            patch.object(online_app, "smps_correction_mode", SimpleNamespace(value="None")),
            patch.object(online_app, "inversion_methods", SimpleNamespace(value=["Gunn-Woessner modified"])),
            patch.object(online_app, "smeariii_sum_root", SimpleNamespace(value=str(root))),
            patch.object(online_app, "load_smeariii_cpc_for_times", side_effect=fake_smear_cpc),
        ):
            result = online_app.run_inversion_calculation(one_scan, use_cache=False)
            figure = online_app.plot_inversion_result(result)
            timing = online_app.plot_smps_timing_diagnostics(result)

        self.assertTrue(any(row["kind"] == "heatmap" for row in result))
        self.assertIsNotNone(figure)
        self.assertIsNotNone(timing)

    def test_smps_heatmap_and_ntot_use_time_not_scan_id_order(self):
        path = Path(__file__).resolve().parents[1] / "SMEARIII" / "DMPS007_20260629.scan"
        data = online_app.parse_smeariii_scan_file(path)
        first, second = data["scan_id"].drop_duplicates().iloc[:2]
        selected = data[data["scan_id"].isin([first, second])].copy()
        # Lexical order puts scan-10 before scan-2 even though it was later.
        selected.loc[selected["scan_id"] == first, "scan_id"] = "scan-2"
        selected.loc[selected["scan_id"] == second, "scan_id"] = "scan-10"

        with (
            patch.object(online_app, "scan_inversion_type", SimpleNamespace(value="SMPS")),
            patch.object(online_app, "smps_correction_mode", SimpleNamespace(value="None")),
            patch.object(online_app, "inversion_methods", SimpleNamespace(value=["Gunn-Woessner modified"])),
        ):
            result = online_app.run_inversion_calculation(selected, use_cache=False)

        heatmap = next(row for row in result if row["kind"] == "heatmap")
        ntot = next(row for row in result if row["kind"] == "ntot")
        self.assertEqual(heatmap["scan_id"], ["scan-2", "scan-10"])
        self.assertEqual(heatmap["x"], ntot["x"])
        self.assertLess(heatmap["x"][0], heatmap["x"][1])

        # Multi-day selections need dates on the time axis, not repeating
        # hour-only labels that hide gaps between scan days.
        next_day = [{**item, "x": [
            pd.Timestamp(time) + pd.Timedelta(days=index)
            for index, time in enumerate(item["x"])
        ]} if item["kind"] in {"heatmap", "ntot"} else item for item in result]
        with (
            patch.object(online_app, "load_smeariii_sum_range", return_value=pd.DataFrame()),
            patch.object(online_app, "load_smeariii_cpc_for_times", return_value=pd.DataFrame()),
        ):
            figure = online_app.plot_inversion_result(next_day)
        self.assertEqual(figure.layout.xaxis.tickformat, "%b %d %H:%M")

    def test_synced_folders_offer_scans_and_reference_sum_separately(self):
        fixture = Path(__file__).resolve().parents[1] / "SMEARIII"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ufsmps = root / "ufsmps"
            smps = root / "smps"
            ufsmps.mkdir()
            smps.mkdir()
            (ufsmps / "UFSMPS_proto_20260629.scan").touch()
            (smps / "DMPS007_20260629.scan").touch()
            copyfile(fixture / "DMPS007_20260629.sum", smps / "DMPS007_20260629.sum")

            with patch.object(online_app, "scan_root", SimpleNamespace(value=str(ufsmps))):
                self.assertEqual(
                    [path.name for path in online_app.list_scan_files()],
                    ["UFSMPS_proto_20260629.scan"],
                )
            with (
                patch.object(online_app, "scan_root", SimpleNamespace(value=str(smps))),
                patch.object(online_app, "smeariii_sum_root", SimpleNamespace(value=str(smps))),
            ):
                self.assertEqual(
                    [path.name for path in online_app.list_scan_files()],
                    ["DMPS007_20260629.scan"],
                )
                comparison = online_app.load_smeariii_sum_range(
                    "2026-06-29 01:02", "2026-06-29 01:05",
                )
                self.assertFalse(comparison.empty)
                self.assertTrue((comparison["smear_conc"] >= 0).any())

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
