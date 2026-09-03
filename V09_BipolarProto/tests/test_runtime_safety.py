import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "dmps_runtime", Path(__file__).parents[1] / "DmpsControl" / "runtime.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeSafetyTests(unittest.TestCase):
    def test_shutdown_coordinator_is_idempotent(self):
        calls = []
        shutdown = runtime.ShutdownCoordinator(calls.append)

        self.assertTrue(shutdown.run("SIGTERM"))
        self.assertFalse(shutdown.run("process exit"))
        self.assertEqual(calls, ["SIGTERM"])
        self.assertTrue(shutdown.started)

    def test_health_json_is_atomic_and_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            runtime.atomic_write_json(
                path,
                {"finite": 1.5, "nan": float("nan"), "nested": [float("inf"), 2]},
            )

            self.assertEqual(
                json.loads(path.read_text()),
                {"finite": 1.5, "nan": None, "nested": [None, 2]},
            )
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
