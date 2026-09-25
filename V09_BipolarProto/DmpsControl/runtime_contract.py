from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class CommandKind(str, Enum):
    INITIALIZE = "initialize"
    START_SCAN = "start_scan"
    STOP_SCAN = "stop_scan"
    SETTING = "setting"
    MANUAL_HV = "manual_hv"
    SNAPSHOT = "snapshot"


class RuntimeState(str, Enum):
    STARTING = "starting"
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    SHUTDOWN = "shutdown"


IDLE_ONLY_COMMANDS = frozenset({
    CommandKind.INITIALIZE,
    CommandKind.SETTING,
    CommandKind.MANUAL_HV,
})


def utc_now():
    return datetime.now(timezone.utc)


def parse_command_kind(value):
    try:
        return CommandKind(value)
    except ValueError as error:
        allowed = ", ".join(command.value for command in CommandKind)
        raise ValueError(f"unknown runtime command {value!r}; expected one of {allowed}") from error


def command_requires_idle(command):
    return parse_command_kind(command) in IDLE_ONLY_COMMANDS


@dataclass(frozen=True)
class RuntimeCommand:
    kind: CommandKind
    requester_id: str
    requester_label: str = "unknown viewer"
    payload: Any = None
    command_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self):
        object.__setattr__(self, "kind", parse_command_kind(self.kind))
        if not str(self.requester_id).strip():
            raise ValueError("requester_id is required")
        object.__setattr__(self, "requester_id", str(self.requester_id))
        object.__setattr__(self, "requester_label", str(self.requester_label or "unknown viewer"))


@dataclass(frozen=True)
class RuntimeSnapshot:
    runtime_id: str
    state: RuntimeState
    scan_active: bool
    status: str
    last_command_id: str | None = None
    last_command_requester: str | None = None
    active_controller_id: str | None = None
    active_controller_label: str | None = None
    updated_at: datetime = field(default_factory=utc_now)
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "state", RuntimeState(self.state))


@dataclass
class ControlLease:
    holder_id: str | None = None
    holder_label: str | None = None
    expires_at: datetime | None = None

    def active(self, now=None):
        now = now or utc_now()
        return self.holder_id is not None and self.expires_at is not None and now < self.expires_at

    def can_accept(self, requester_id, now=None):
        now = now or utc_now()
        return not self.active(now) or self.holder_id == str(requester_id)

    def claim(self, requester_id, requester_label, ttl_seconds=30, now=None):
        now = now or utc_now()
        if not self.can_accept(requester_id, now):
            return False
        self.holder_id = str(requester_id)
        self.holder_label = str(requester_label or "unknown viewer")
        self.expires_at = now + timedelta(seconds=float(ttl_seconds))
        return True

    def release(self, requester_id):
        if self.holder_id != str(requester_id):
            return False
        self.holder_id = None
        self.holder_label = None
        self.expires_at = None
        return True


def validate_command_allowed(command, snapshot):
    command = command if isinstance(command, RuntimeCommand) else RuntimeCommand(**command)
    snapshot = snapshot if isinstance(snapshot, RuntimeSnapshot) else RuntimeSnapshot(**snapshot)
    if snapshot.state in {RuntimeState.SHUTDOWN, RuntimeState.ERROR}:
        return False, f"runtime is {snapshot.state.value}"
    if command_requires_idle(command.kind) and snapshot.scan_active:
        return False, f"{command.kind.value} is only allowed while measurement is idle"
    return True, "accepted"
