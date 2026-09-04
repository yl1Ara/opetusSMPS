import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "dmps_runtime", Path(__file__).parents[1] / "DmpsControl" / "runtime.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

idle_spec = importlib.util.spec_from_file_location(
    "check_measurement_idle",
    Path(__file__).parents[1] / "deploy" / "check-measurement-idle.py",
)
check_measurement_idle = importlib.util.module_from_spec(idle_spec)
idle_spec.loader.exec_module(check_measurement_idle)

class RuntimeSafetyTests(unittest.TestCase):
    def test_gui_refresh_uses_single_owner_bridge_without_stopping_previous_runtime(self):
        root = Path(__file__).parents[1]
        source = (root / "gui_app.py").read_text()
        entrypoint = (root / "gui.py").read_text()
        runtime_host = (root / "gui_runtime_host.py").read_text()

        self.assertIn('RUNTIME_BRIDGE_KEY = "tdmps_gui_runtime_bridge"', source)
        self.assertNotIn("previous_app_stop_event.set()", source)
        self.assertNotIn("previous_flow_controller.stop()", source)
        self.assertIn('if runtime_owner:', source)
        self.assertNotIn("runtime_command_thread", source)
        self.assertIn("owner_document.add_next_tick_callback", source)
        self.assertIn("owner_document.add_periodic_callback(drain_ui_updates", source)
        self.assertIn("serve_gui", entrypoint)
        self.assertIn("_OWNER_NAMESPACE = namespace", runtime_host)
        self.assertIn("pn.state.on_session_destroyed(runtime_owner_session_destroyed)", source)
        run_panel = (root / "deploy" / "run-panel.sh").read_text()
        self.assertIn("--reuse-sessions", run_panel)

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

    def test_measurement_idle_check_reads_health_without_opening_panel_session(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            base = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "runtime_id": "runtime-a", "pid": 123,
                "runtime_state": "idle", "scan_active": False,
            }
            path.write_text(json.dumps(base))
            self.assertEqual(check_measurement_idle.measurement_state(path)[0], 0)
            self.assertEqual(
                check_measurement_idle.measurement_state(path, expected_pid=123)[0],
                0,
            )

            with self.assertRaises(ValueError):
                check_measurement_idle.measurement_state(path, expected_pid=456)

            path.write_text(json.dumps({
                **base, "runtime_state": "running", "scan_active": True,
            }))
            self.assertEqual(check_measurement_idle.measurement_state(path)[0], 10)

    def test_measurement_idle_check_fails_closed_for_stale_health(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            path.write_text(json.dumps({
                "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                "runtime_id": "runtime-a", "pid": 123,
                "runtime_state": "idle", "scan_active": False,
            }))

            with self.assertRaises(ValueError):
                check_measurement_idle.measurement_state(path, maximum_age_seconds=10)


if __name__ == "__main__":
    unittest.main()
