from GP8XXX_IIC import GP8403
import serial
import time
import smbus2
import threading
from smbus2 import i2c_msg
from gpiozero import OutputDevice, PWMOutputDevice

class CPC:
    def __init__(self, port, CPC_type="3010"):
        self.type = CPC_type
        self.lock = threading.Lock()
        self.terminator = b"\r"
        if self.type == "3010":
            self.ser = serial.Serial(
                port=port,
                baudrate=9600,
                bytesize=serial.SEVENBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
                write_timeout=0.2,
            )
            self.concentration_command = "RD"
        elif self.type == "3771":
            self.ser = serial.Serial(
                port=port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
                write_timeout=0.2,
            )
            self.concentration_command = "RD"
        elif self.type == "HY09":
            self.ser = serial.Serial(
                port=port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
                write_timeout=0.2,
            )
            self.concentration_command = "RB"
        else:
            raise ValueError(f"Unsupported CPC type: {CPC_type}")

    def _read_record(self):
        return self.ser.read_until(self.terminator).decode("ascii", errors="replace").strip()

    def read_instrument(self):
        response = self.query(self.concentration_command)
        return response if response else "nan"

    def query(self, command, read_lines=1, reset_input=True):
        cmd = str(command).strip()
        if not cmd:
            raise ValueError("CPC command is empty")

        with self.lock:
            if reset_input:
                self.ser.reset_input_buffer()
            payload = cmd.encode("ascii", errors="replace") + self.terminator
            self.ser.write(payload)
            self.ser.flush()

            lines = []
            echo_skipped = False
            while len(lines) < max(1, int(read_lines)):
                line = self._read_record()
                if not echo_skipped and line.casefold() == cmd.casefold():
                    echo_skipped = True
                    continue
                if line.upper() in {"ERROR", "SWITCH ERROR"}:
                    raise RuntimeError(f"CPC rejected {cmd}: {line}")
                lines.append(line)
                if not line:
                    break

        return "\n".join(lines).strip()

    def close(self):
        with self.lock:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()


class HaukeDMA:
    def __init__(self):
        self.r1 = 0.025
        self.r2 = 0.033
        self.L = 0.28
        
I2C_BUS = 1
I2C_ADDRESS = 0x40

SCALE_FACTOR = 140.0
OFFSET = 32000.0

def crc8(data):
    # SFM3xxx CRC-8: polynomial 0x31, initial value 0x00.
    crc = 0x00

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc


class SFM3000CRCError(RuntimeError):
    pass


class Flowmeter:
    def __init__(self, bus=I2C_BUS, address=I2C_ADDRESS, bus_factory=smbus2.SMBus, max_errors=5):
        self.bus_number = int(bus)
        self.address = int(address)
        self.bus_factory = bus_factory
        self.max_errors = max(1, int(max_errors))
        self.bus = None
        self.flow = 0.0
        self.sample_monotonic = None
        self.connected = False
        self.error_count = 0
        self.consecutive_errors = 0
        self.crc_error_count = 0
        self.reconnect_count = 0
        self.last_error = None
        self.serial_number = None
        self.article_number = None
        self.scale_factor = SCALE_FACTOR
        self.offset = OFFSET
        self._lock = threading.RLock()
        self._connect(query_identity=True)

    def _connect(self, query_identity=True):
        self._close_bus()
        self.bus = self.bus_factory(self.bus_number)
        try:
            if query_identity:
                self.serial_number = self.read_u32(0x31AE, 0x31AF)
                self.article_number = self.read_u32(0x31E3, 0x31E4)
                scale = self.read_signed_word(0x30DE)
                offset = self.read_signed_word(0x30DF)
                if scale <= 0:
                    raise RuntimeError(f"SFM3000 invalid scale factor {scale}")
                self.scale_factor = float(scale)
                self.offset = float(offset)
            self.start_measurement()
        except Exception:
            self._close_bus()
            raise
        self.connected = True
        self.consecutive_errors = 0
        self.last_error = None
        print(
            f"SFM3000 connected: serial={self.serial_number}, article={self.article_number}, "
            f"scale={self.scale_factor:g}, offset={self.offset:g}",
            flush=True,
        )

    def _close_bus(self):
        bus, self.bus = self.bus, None
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass

    def write_command(self, command):
        msb = (command >> 8) & 0xFF
        lsb = command & 0xFF

        if self.bus is None:
            raise RuntimeError("SFM3000 I2C bus is closed")
        msg = i2c_msg.write(self.address, [msb, lsb])
        self.bus.i2c_rdwr(msg)

    def read_word(self):
        if self.bus is None:
            raise RuntimeError("SFM3000 I2C bus is closed")
        msg = i2c_msg.read(self.address, 3)
        self.bus.i2c_rdwr(msg)

        data = list(msg)

        msb = data[0]
        lsb = data[1]
        crc = data[2]

        if crc8(bytes([msb, lsb])) != crc:
            raise SFM3000CRCError(
                f"SFM3000 CRC mismatch: received 0x{crc:02x} for {msb:02x}{lsb:02x}"
            )

        return (msb << 8) | lsb

    def read_signed_word(self, command):
        self.write_command(command)
        time.sleep(0.001)
        value = self.read_word()
        return value - 0x10000 if value >= 0x8000 else value

    def read_u32(self, high_command, low_command):
        self.write_command(high_command)
        time.sleep(0.001)
        high = self.read_word()
        self.write_command(low_command)
        time.sleep(0.001)
        return (high << 16) | self.read_word()

    def start_measurement(self):
        last_error = None
        for i in range(5):
            try:
                self.write_command(0x1000)
                time.sleep(0.1)
                return
            except OSError as e:
                last_error = e
                print(f"SFM3000 start failed {i+1}/5: {e}", flush=True)
                time.sleep(0.2)
        raise RuntimeError("SFM3000 did not start") from last_error

    def step(self):
        with self._lock:
            try:
                raw = self.read_word()
                self.flow = (raw - self.offset) / self.scale_factor
                self.sample_monotonic = time.monotonic()
                self.connected = True
                self.consecutive_errors = 0
                self.last_error = None
            except Exception as error:
                self.error_count += 1
                self.consecutive_errors += 1
                self.last_error = str(error)
                if isinstance(error, SFM3000CRCError):
                    self.crc_error_count += 1
                if self.consecutive_errors >= self.max_errors:
                    self.connected = False
                    self.reconnect_count += 1
                    try:
                        self._connect(query_identity=False)
                        self.last_error = f"{error}; reconnected"
                    except Exception as reconnect_error:
                        self.last_error = f"{error}; reconnect failed: {reconnect_error}"
                raise

    def get_flow(self):
        with self._lock:
            return self.flow

    def get_sample(self):
        with self._lock:
            return self.flow, self.sample_monotonic

    def diagnostics(self):
        with self._lock:
            sample_age = (
                time.monotonic() - self.sample_monotonic
                if self.sample_monotonic is not None
                else None
            )
            return {
                "connected": self.connected,
                "sample_age_sec": sample_age,
                "error_count": self.error_count,
                "consecutive_errors": self.consecutive_errors,
                "crc_error_count": self.crc_error_count,
                "reconnect_count": self.reconnect_count,
                "last_error": self.last_error,
                "serial_number": self.serial_number,
                "article_number": self.article_number,
                "scale_factor": self.scale_factor,
                "offset": self.offset,
            }

    def close(self):
        with self._lock:
            self.connected = False
            self._close_bus()


class AerosolFlowmeter:
    """Cached SDP8xx pressure, temperature, and calibrated aerosol flow."""

    def __init__(self, bus=1, address=0x25, calibration_lpm_per_pa=0.15228426395939088):
        # Keep this optional hardware dependency out of normal startup/imports.
        from sensirion_i2c_driver import I2cConnection, LinuxI2cTransceiver
        from sensirion_i2c_sdp import SdpI2cDevice

        transceiver = LinuxI2cTransceiver(f"/dev/i2c-{int(bus)}")
        self.transceiver = transceiver
        connection = I2cConnection(transceiver)
        self.device = SdpI2cDevice(connection, slave_address=int(address))
        self.calibration = float(calibration_lpm_per_pa)
        self.lock = threading.Lock()
        self.io_lock = threading.Lock()
        self.pressure_pa = float("nan")
        self.temperature_c = float("nan")
        try:
            self.device.stop_continuous_measurement()
        except Exception:
            pass
        self.device.start_continuous_measurement_with_mass_flow_t_comp()
        time.sleep(0.01)
        self.step()

    def step(self):
        with self.io_lock:
            pressure, temperature = self.device.read_measurement()
        with self.lock:
            self.pressure_pa = float(pressure.pascal)
            self.temperature_c = float(temperature.degrees_celsius)

    def snapshot(self):
        with self.lock:
            pressure = self.pressure_pa
            temperature = self.temperature_c
        return self.calibration * pressure, pressure, temperature

    def read_pressure_samples(self, count=10, interval_sec=0.5, cancel_event=None):
        pressures = []
        for index in range(max(1, int(count))):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Aerosol calibration cancelled")
            self.step()
            _, pressure, _ = self.snapshot()
            pressures.append(pressure)
            if index + 1 < count:
                time.sleep(max(0.05, float(interval_sec)))
        return pressures

    def close(self):
        try:
            with self.io_lock:
                self.device.stop_continuous_measurement()
        except Exception:
            pass
        close = getattr(self.transceiver, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


class PicoValve:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=2)
        time.sleep(2)
        self.ser.reset_input_buffer()

    def command(self, cmd):
        self.ser.write((cmd + "\n").encode())
        return self.ser.readline().decode().strip()

    def on(self):
        return self.command("ON")

    def off(self):
        return self.command("OFF")

    def status(self):
        return self.command("STATUS")

    def close(self):
        self.off()
        self.ser.close()
        
class DacOut:
    def __init__(self, pin=18):
        self.sw = OutputDevice(pin, initial_value=0)

    def allow(self):
        self.sw.value = 1

    def block(self):
        self.sw.value = 0

    def close(self):
        self.block()
        self.sw.close()


class BlowerDAC:
    def __init__(self):
        self.dac = GP8403(
            i2c_addr=0x5F,
            bus=1
        )

        self.voltage = 0.0

        # test communication
        self.dac.set_dac_out_voltage(
            voltage=0,
            channel=1
        )

        print("GP8403 connected")

    def set_voltage(self, voltage):
        self.voltage = float(voltage)

        # limit 0–5V
        self.voltage = max(0, min(5, self.voltage))

        self.dac.set_dac_out_voltage(
            voltage=self.voltage,
            channel=1
        )

    def get_parameter(self):
        return self.voltage

    def close(self):
        self.set_voltage(0.0)
        close = getattr(self.dac, "close", None)
        if close is not None:
            close()

if __name__ == "__main__":
    valve = PicoValve()
    print("Turning valve ON...")
    print(valve.on())
    time.sleep(20)
    print("Valve status:", valve.status())
    print("Turning valve OFF...")
    print(valve.off())
    time.sleep(2)
    print("Valve status:", valve.status())
    valve.close()
