import unittest
import importlib.util
from datetime import timedelta
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "runtime_contract", Path(__file__).parents[1] / "DmpsControl" / "runtime_contract.py"
)
runtime_contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime_contract)

CommandKind = runtime_contract.CommandKind
ControlLease = runtime_contract.ControlLease
RuntimeCommand = runtime_contract.RuntimeCommand
RuntimeSnapshot = runtime_contract.RuntimeSnapshot
RuntimeState = runtime_contract.RuntimeState
command_requires_idle = runtime_contract.command_requires_idle
utc_now = runtime_contract.utc_now
validate_command_allowed = runtime_contract.validate_command_allowed


class RuntimeContractTests(unittest.TestCase):
    def test_browser_sessions_do_not_define_runtime_state(self):
        snapshot = RuntimeSnapshot(
            runtime_id="dmps-01-runtime",
            state=RuntimeState.RUNNING,
            scan_active=True,
            status="scan 4/20",
            active_controller_id="viewer-a",
            active_controller_label="student group A",
        )

        self.assertTrue(snapshot.scan_active)
        self.assertEqual(snapshot.active_controller_label, "student group A")
        self.assertNotIn("session", snapshot.__dataclass_fields__)

    def test_only_one_viewer_holds_control_lease_at_a_time(self):
        now = utc_now()
        lease = ControlLease()

        self.assertTrue(lease.claim("viewer-a", "Alice", now=now))
        self.assertTrue(lease.can_accept("viewer-a", now=now + timedelta(seconds=1)))
        self.assertFalse(lease.can_accept("viewer-b", now=now + timedelta(seconds=1)))
        self.assertTrue(lease.can_accept("viewer-b", now=now + timedelta(seconds=31)))

    def test_idle_only_commands_are_rejected_during_scan(self):
        running = RuntimeSnapshot(
            runtime_id="dmps-01-runtime",
            state=RuntimeState.RUNNING,
            scan_active=True,
            status="scan running",
        )
        command = RuntimeCommand(
            kind=CommandKind.SETTING,
            requester_id="viewer-a",
            requester_label="Alice",
            payload=("Measurement time", 5),
        )

        accepted, reason = validate_command_allowed(command, running)

        self.assertFalse(accepted)
        self.assertIn("idle", reason)

    def test_stop_remains_available_during_scan(self):
        running = RuntimeSnapshot(
            runtime_id="dmps-01-runtime",
            state=RuntimeState.RUNNING,
            scan_active=True,
            status="scan running",
        )
        command = RuntimeCommand(
            kind=CommandKind.STOP_SCAN,
            requester_id="viewer-a",
            requester_label="Alice",
        )

        self.assertFalse(command_requires_idle(CommandKind.STOP_SCAN))
        self.assertEqual(validate_command_allowed(command, running), (True, "accepted"))


if __name__ == "__main__":
    unittest.main()
