import importlib.util
import json
import socket
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


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

port_spec = importlib.util.spec_from_file_location(
    "check_port_available",
    Path(__file__).parents[1] / "deploy" / "check-port-available.py",
)
check_port_available = importlib.util.module_from_spec(port_spec)
port_spec.loader.exec_module(check_port_available)

health_log_spec = importlib.util.spec_from_file_location(
    "log_system_health",
    Path(__file__).parents[1] / "deploy" / "log-system-health.py",
)
log_system_health = importlib.util.module_from_spec(health_log_spec)
health_log_spec.loader.exec_module(log_system_health)

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
        self.assertIn('"runtime_heartbeat"', source)
        self.assertIn("fail_measurement(e, _traceback.format_exc())", source)
        self.assertNotIn("runtime_command_thread", source)
        self.assertIn("owner_document.add_next_tick_callback", source)
        self.assertIn("owner_document.add_periodic_callback(drain_ui_updates", source)
        self.assertIn("serve_gui", entrypoint)
        self.assertIn("_OWNER_NAMESPACE = namespace", runtime_host)
        self.assertIn("pn.state.on_session_destroyed(runtime_session_destroyed)", source)
        run_panel = (root / "deploy" / "run-panel.sh").read_text()
        self.assertIn("--reuse-sessions", run_panel)

    def test_shutdown_coordinator_is_idempotent(self):
        calls = []
        shutdown = runtime.ShutdownCoordinator(calls.append)

        self.assertTrue(shutdown.run("SIGTERM"))
        self.assertFalse(shutdown.run("process exit"))
        self.assertEqual(calls, ["SIGTERM"])
        self.assertTrue(shutdown.started)

    def test_runtime_event_log_is_durable_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            log = runtime.RuntimeEventLog(Path(directory), "test")
            self.assertTrue(log.write("measurement_started", value=float("nan")))
            path = next(Path(directory).glob("runtime-events-*.jsonl"))
            event = json.loads(path.read_text())

            self.assertEqual(event["event"], "measurement_started")
            self.assertEqual(event["component"], "test")
            self.assertIsNone(event["value"])
            self.assertIn("boot_id", event)

    def test_port_guard_detects_existing_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            self.assertFalse(check_port_available.port_available("127.0.0.1", port))
        self.assertTrue(check_port_available.port_available("127.0.0.1", port))

    def test_service_conflicts_with_legacy_and_checks_port_before_hardware(self):
        service = (
            Path(__file__).parents[1] / "deploy" / "tdmps@.service"
        ).read_text()
        self.assertIn("Conflicts=tdmps.service", service)
        self.assertIn("StartLimitBurst=3", service)
        self.assertIn("${DMPS_PANEL_PORT}", service)
        self.assertIn("MemoryHigh=768M", service)
        self.assertIn("MemoryMax=1G", service)
        self.assertIn("MemorySwapMax=256M", service)
        self.assertLess(service.index("ExecCondition="), service.index("ExecStartPre="))
        journal_config = (
            Path(__file__).parents[1] / "deploy" / "60-tdmps-persistent-journal.conf"
        ).read_text()
        self.assertIn("Storage=persistent", journal_config)
        force_safe = (
            Path(__file__).parents[1] / "deploy" / "run-force-safe.sh"
        ).read_text()
        self.assertIn('SERVICE_RESULT:-}" == "exec-condition"', force_safe)
        for script_name in ("dmps", "install-services.sh"):
            script = (Path(__file__).parents[1] / "deploy" / script_name).read_text()
            self.assertIn('${HOME}/.local/bin:${PATH}', script)

    def test_system_health_log_is_persistent_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            record = {
                "timestamp": "2026-09-21T00:00:00+00:00",
                "cpu_temperature_c": 64.5,
                "throttling": {"raw": "0x0"},
            }
            path = log_system_health.append_record(Path(directory), record)

            self.assertEqual(json.loads(path.read_text()), record)
            self.assertEqual(path.name, "system-health-20260921.jsonl")

    def test_cpu_temperature_uses_cpu_thermal_zone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "thermal_zone0").mkdir()
            (root / "thermal_zone0" / "type").write_text("cpu-thermal\n")
            (root / "thermal_zone0" / "temp").write_text("81500\n")

            self.assertEqual(log_system_health.cpu_temperature_celsius(root), 81.5)

    def test_throttling_flags_include_historical_soft_temperature_limit(self):
        with mock.patch.object(log_system_health.subprocess, "run") as run:
            run.return_value.stdout = "throttled=0x80000\n"

            status = log_system_health.throttling_status()

        self.assertTrue(status["soft_temperature_limit_occurred"])
        self.assertFalse(status["soft_temperature_limit_now"])

    def test_system_health_service_is_installed_and_enabled(self):
        root = Path(__file__).parents[1]
        service = (root / "deploy" / "tdmps-health-log@.service").read_text()
        self.assertIn("DMPS_STATE_DIR", service)
        self.assertIn("Restart=on-failure", service)
        for script_name in ("dmps", "install-services.sh"):
            script = (root / "deploy" / script_name).read_text()
            self.assertIn("tdmps-health-log@", script)
            self.assertIn("dmps-log-system-health", script)

    def test_reboot_diagnostics_include_kernel_oom_records(self):
        script = (Path(__file__).parents[1] / "deploy" / "dmps").read_text()

        self.assertIn("[Oo]ut of memory", script)
        self.assertIn("oom-killer", script)
        self.assertIn("Killed process", script)

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
