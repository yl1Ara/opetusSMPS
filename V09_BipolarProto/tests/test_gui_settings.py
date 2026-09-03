import ast
import unittest
from pathlib import Path


class GuiSettingsTests(unittest.TestCase):
    def test_active_scan_snapshot_contains_step_shift(self):
        source = (Path(__file__).parents[1] / "gui.py").read_text()
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


if __name__ == "__main__":
    unittest.main()
