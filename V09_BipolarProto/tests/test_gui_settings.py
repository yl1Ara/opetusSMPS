import ast
import unittest
from pathlib import Path

import numpy as np


GUI_PATH = Path(__file__).parents[1] / "gui_app.py"


def load_timing_helpers():
    module = ast.parse(GUI_PATH.read_text())
    wanted = {
        "robust_cpc_cadence_seconds", "cpc_response_window_seconds",
        "cpc_scan_timing_profile",
    }
    nodes = [
        node for node in module.body
        if (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in {
                "DAILY_MEASUREMENT_COLUMNS", "CPC_NOMINAL_CADENCE_SEC"
            } for target in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in wanted)
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(GUI_PATH), "exec"), namespace)
    return namespace


class GuiSettingsTests(unittest.TestCase):
    def test_active_scan_snapshot_contains_step_shift(self):
        source = GUI_PATH.read_text()
        module = ast.parse(source)
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "current_scan_settings"
        )
        returned = next(
            node.value for node in ast.walk(function) if isinstance(node, ast.Return)
        )
        keys = {
            key.value
            for key in returned.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

        self.assertIn("smps_plot_step_shift", keys)

    def test_active_scan_snapshot_contains_all_cpc_timing_inputs(self):
        source = GUI_PATH.read_text()
        module = ast.parse(source)
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "current_scan_settings"
        )
        returned = next(node.value for node in ast.walk(function) if isinstance(node, ast.Return))
        keys = {key.value for key in returned.keys if isinstance(key, ast.Constant)}

        self.assertTrue({
            "cpc_type", "meas_time", "cpc_poll_interval",
            "cpc_transport_delay_sec", "cpc_response_window_sec",
            "automatic_boundary_holds", "initial_point_pre_hold",
            "final_point_extra_hold",
        }.issubset(keys))

    def test_per_scan_rows_have_self_describing_timing_and_software_fields(self):
        module = ast.parse(GUI_PATH.read_text())
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "append_measurement_row"
        )
        row_assignment = next(
            node for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "row" for target in node.targets)
        )
        keys = {key.value for key in row_assignment.value.keys if isinstance(key, ast.Constant)}

        self.assertTrue({
            "cpc_type", "point_dwell_sec", "cpc_poll_interval_sec",
            "cpc_transport_delay_sec", "cpc_response_window_sec",
            "measured_cpc_cadence_sec", "boundary_hold_policy",
            "initial_pre_hold_sec", "final_hold_sec", "point_index",
            "point_set_time", "point_valid_from", "point_valid_until",
            "cpc_response_window_rule", "acquisition_session_id",
            "cpc_query_start_time", "cpc_query_end_time",
            "cpc_timestamp_uncertainty_sec", "cpc_timestamp_basis",
            "app_version", "git_commit",
        }.issubset(keys))


class CpcScanTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers = load_timing_helpers()
        cls.profile = staticmethod(helpers["cpc_scan_timing_profile"])
        cls.cadence = staticmethod(helpers["robust_cpc_cadence_seconds"])
        cls.response_window = staticmethod(helpers["cpc_response_window_seconds"])
        cls.daily_columns = helpers["DAILY_MEASUREMENT_COLUMNS"]

    def test_one_second_3010_profile_meets_two_sample_target(self):
        profile = self.profile("3010", 2, 0.5, 3, 2, True, 9, 10, 1.0)

        self.assertEqual(profile["expected_unique_samples_per_point"], 2)
        self.assertEqual(profile["target_unique_samples_per_point"], 2)
        self.assertFalse(profile["sample_count_warning"])
        self.assertEqual(profile["initial_pre_hold_sec"], 10.0)
        self.assertEqual(profile["final_hold_sec"], 10.0)

    def test_one_second_3771_profile_meets_one_sample_target(self):
        profile = self.profile("3771", 1, 0.5, 3, 9, True, 9, 10, 1.018)

        self.assertEqual(profile["expected_unique_samples_per_point"], 1)
        self.assertEqual(profile["target_unique_samples_per_point"], 1)
        self.assertFalse(profile["sample_count_warning"])
        self.assertAlmostEqual(profile["final_hold_sec"], 5.018)

    def test_polling_faster_than_cpc_reporting_does_not_inflate_sample_count(self):
        profile = self.profile("3771", 1, 0.1, 3, 1, True, 9, 10, 0.25)

        self.assertEqual(profile["expected_unique_samples_per_point"], 1)
        self.assertEqual(profile["cadence_margin_sec"], 1.0)

    def test_3010_response_window_tracks_reported_concentration(self):
        self.assertEqual(self.response_window("3010", 99.9, 2), (6.0, "3010 concentration <= 100"))
        self.assertEqual(self.response_window("3010", 100, 2), (6.0, "3010 concentration <= 100"))
        self.assertEqual(self.response_window("3010", 100.1, 2), (1.0, "3010 concentration > 100"))
        self.assertEqual(self.response_window("3771", 10, 9), (1.0, "3771 documented 1 s average"))

    def test_new_timing_settings_do_not_enable_automatic_holds_by_default(self):
        source = GUI_PATH.read_text()
        module = ast.parse(source)
        defaults = next(
            node.value for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "DEFAULT_SETTINGS" for target in node.targets)
        )
        values = {
            key.value: value.value
            for key, value in zip(defaults.keys, defaults.values)
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
        }
        self.assertFalse(values["automatic_boundary_holds"])

    def test_cadence_uses_unique_ids_and_median_interval(self):
        samples = [(1, 10.0), (1, 99.0), (2, 11.0), (3, 13.0), (4, 14.0)]
        self.assertEqual(self.cadence(samples), 1.0)

    def test_manual_holds_are_not_recomputed(self):
        profile = self.profile("HY09", 1, 0.5, 30, 5, False, 4, 7, 2.0)
        self.assertEqual(profile["hold_policy"], "manual")
        self.assertEqual(profile["initial_pre_hold_sec"], 4.0)
        self.assertEqual(profile["final_hold_sec"], 7.0)
        self.assertTrue(profile["sample_count_warning"])

    def test_daily_measurement_schema_remains_exactly_fifteen_columns(self):
        self.assertEqual(self.daily_columns, (
            "time", "scan_range", "size_nm", "cpc_count", "sheath_flow",
            "sheath_setpoint", "scan_number", "Ntot", "sample_duration_sec",
            "cpc_age_sec", "cpc_read_duration_sec", "cpc_error",
            "point_elapsed_sec", "point_set_duration_sec", "phase",
        ))


if __name__ == "__main__":
    unittest.main()
