if __name__ == "__main__":
    from hardware import BlowerDAC, Flowmeter
    from tuning import lambda_pi_from_step
else:
    from .hardware import BlowerDAC, Flowmeter
    from .tuning import lambda_pi_from_step
from simple_pid import PID
import threading
import time
import math
from datetime import datetime



class FlowController:
    def __init__(self, flowmeter, blower, flow_lpm=10, kp=0.008, ki=0.015, kd=0.0):
        self.flowmeter = flowmeter
        self.blower = blower
        self.pid = PID(kp, ki, kd, setpoint=flow_lpm, auto_mode=False)
        self.pid.output_limits = (0, 5)
        self.running = False
        self.thread = None
        self._io_lock = threading.Lock()
        self._tuning_lock = threading.Lock()
        self._tuning = threading.Event()
        self._shutdown = threading.Event()
        self.flow_error_count = 0
        self.out = 2.5
        self.blower.set_voltage(self.out)
        self.pid.set_auto_mode(True, last_output=self.out)

    def setpoint(self, flow_lpm):
        with self._io_lock:
            self.pid.setpoint = flow_lpm
            if flow_lpm <= 0:
                self.out = 0.0
                self.pid.reset()
                self.blower.set_voltage(self.out)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True, name="sheath-flow-controller")
        self.thread.start()

    def stop(self, timeout=2.0):
        self.running = False
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=timeout)

    def emergency_stop(self):
        self.running = False
        self._shutdown.set()
        self._tuning.set()
        with self._io_lock:
            self.pid.setpoint = 0.0
            self.pid.set_auto_mode(False)
            self.out = 0.0
            self.blower.set_voltage(0.0)

    def loop(self):
        while self.running:
            if self._tuning.is_set():
                time.sleep(0.05)
                continue
            try:
                with self._io_lock:
                    self.flowmeter.step()
                    raw = self.flowmeter.get_flow()
                    setpoint = float(self.pid.setpoint)
                    if setpoint <= 0:
                        self.out = 0.0
                    else:
                        self.out = self.pid(raw)
                    self.blower.set_voltage(self.out)
                    self.flow_error_count = 0
            except Exception as e:
                self.flow_error_count += 1
                print(f"Flowmeter error {self.flow_error_count}: {e}", flush=True)
                with self._io_lock:
                    self.out = 0.0
                    self.blower.set_voltage(self.out)
                print("Flowmeter failure: blower commanded to zero", flush=True)
                time.sleep(0.5)

            time.sleep(0.1)

    def pid_params(self):
        return {"Kp": self.pid.Kp, "Ki": self.pid.Ki, "Kd": self.pid.Kd}

    def set_pid_params(self, kp, ki, kd):
        values = (float(kp), float(ki), float(kd))
        if values[0] <= 0 or values[1] <= 0 or values[2] < 0:
            raise ValueError("PID gains must have Kp > 0, Ki > 0, and Kd >= 0")
        with self._io_lock:
            self.pid.tunings = values
            self.pid.reset()

    def run_step_tuning(self, config, cancel_event, sample_callback=None, progress_callback=None):
        """Temporarily own the real sheath loop and collect an open-loop DAC step."""
        if not isinstance(self.flowmeter, Flowmeter):
            raise RuntimeError("Tuning requires the real sheath SFM3000 Flowmeter")
        if not self.running:
            raise RuntimeError("Sheath flow controller is not running")
        low = float(config["output_low_v"])
        high = float(config["output_high_v"])
        settle = float(config["settle_sec"])
        step_sec = float(config["step_sec"])
        interval = float(config["sample_interval_sec"])
        if not (0.0 <= low < high <= 5.0) or high - low < 0.2:
            raise ValueError("Use 0..5 V outputs with a step of at least 0.2 V")
        if not 0.05 <= interval <= 5.0:
            raise ValueError("Sample interval must be 0.05..5 s")
        if settle < max(2.0, 5.0 * interval) or step_sec < max(2.0, 5.0 * interval):
            raise ValueError("Each plateau must allow at least five samples and last at least 2 s")
        if not self._tuning_lock.acquire(blocking=False):
            raise RuntimeError("Sheath PID tuning is already running")

        previous_setpoint = None
        previous_output = None
        samples = []
        self._tuning.set()
        try:
            with self._io_lock:
                previous_setpoint = float(self.pid.setpoint)
                previous_output = float(self.out)
                self.pid.set_auto_mode(False)
            started = time.monotonic()
            for phase, output, duration in (("low", low, settle), ("high", high, step_sec)):
                phase_started = time.monotonic()
                with self._io_lock:
                    self.blower.set_voltage(output)
                    self.out = output
                while time.monotonic() - phase_started < duration:
                    if cancel_event.is_set():
                        raise RuntimeError("Sheath PID tuning cancelled")
                    with self._io_lock:
                        self.flowmeter.step()
                        flow, sample_time = self.flowmeter.get_sample()
                    now = time.monotonic()
                    if sample_time is None or now - sample_time > max(0.5, 2.0 * interval):
                        raise RuntimeError("Stale sheath-flow sample during tuning")
                    row = {
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_sec": now - started,
                        "step_elapsed_sec": now - phase_started if phase == "high" else 0.0,
                        "phase": phase,
                        "output_v": output,
                        "flow_lpm": float(flow),
                    }
                    samples.append(row)
                    if sample_callback:
                        sample_callback(row)
                    if not math.isfinite(row["flow_lpm"]):
                        raise RuntimeError("Nonfinite sheath-flow sample during tuning")
                    if not config["flow_min_lpm"] <= row["flow_lpm"] <= config["flow_max_lpm"]:
                        raise RuntimeError("Out-of-range sheath-flow sample during tuning")
                    if progress_callback:
                        total = settle + step_sec
                        progress_callback(min((now - started) / total, 0.99), phase, float(flow))
                    if cancel_event.wait(interval):
                        raise RuntimeError("Sheath PID tuning cancelled")

            result = lambda_pi_from_step(
                samples, low, high,
                min_response=float(config["min_response_lpm"]),
                flow_min=float(config["flow_min_lpm"]),
                flow_max=float(config["flow_max_lpm"]),
            )
            result["samples"] = samples
            return result
        finally:
            try:
                if self._shutdown.is_set():
                    with self._io_lock:
                        self.pid.setpoint = 0.0
                        self.pid.set_auto_mode(False)
                        self.out = 0.0
                        self.blower.set_voltage(0.0)
                elif previous_setpoint is not None and previous_output is not None:
                    with self._io_lock:
                        self.pid.setpoint = previous_setpoint
                        self.out = previous_output
                        restore_errors = []
                        try:
                            self.blower.set_voltage(previous_output)
                        except Exception as error:
                            restore_errors.append(f"DAC output: {error}")
                        try:
                            self.pid.reset()
                            self.pid.set_auto_mode(True, last_output=previous_output)
                        except Exception as error:
                            restore_errors.append(f"closed-loop PID: {error}")
                        if restore_errors:
                            raise RuntimeError("Failed to restore " + "; ".join(restore_errors))
            finally:
                self._tuning.clear()
                self._tuning_lock.release()


if __name__ == "__main__":
    flowmeter = Flowmeter()
    blower = BlowerDAC()

    controller = FlowController(flowmeter, blower, 10)
    controller.start()

    while True:
        time.sleep(1)
