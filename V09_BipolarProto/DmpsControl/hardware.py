from GP8XXX_IIC import GP8403
import serial
import time
import smbus2
import threading
from smbus2 import i2c_msg
from gpiozero import OutputDevice, PWMOutputDevice

class CPC:
    def __init__(self, port, CPC_type="3771"):
        self.type = CPC_type
        self.lock = threading.Lock()
        if CPC_type == "3771":
            self.ser = serial.Serial(
                port=port,
                baudrate=9600,
                bytesize=serial.SEVENBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
                write_timeout=0.2,
            )
        elif CPC_type == "HY09":
            self.ser = serial.Serial(
                port=port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
                write_timeout=0.2,
            )

    def read_instrument(self):
        with self.lock:
            self.ser.reset_input_buffer()
            if self.type == "3771":
                self.ser.write(b"RD\r")
            elif self.type == "HY09":
                self.ser.write(b"RB\r")
            line = self.ser.readline().decode("utf-8", errors="replace").strip()
        return line if line else "nan"

    def query(self, command, read_lines=1, reset_input=True):
        cmd = str(command).strip()
        if not cmd:
            raise ValueError("CPC command is empty")

        with self.lock:
            if reset_input:
                self.ser.reset_input_buffer()
            payload = cmd.encode("ascii", errors="replace")
            if not payload.endswith((b"\r", b"\n")):
                payload += b"\r"
            self.ser.write(payload)
            self.ser.flush()

            lines = []
            for _ in range(max(1, int(read_lines))):
                line = self.ser.readline().decode("utf-8", errors="replace").strip()
                lines.append(line)
                if not line:
                    break

        return "\n".join(lines).strip()


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
    crc = 0x00

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc


class Flowmeter:
    def __init__(self):
        self.bus = smbus2.SMBus(I2C_BUS)
        self.flow = 0.0
        self.sample_monotonic = None

        self.start_measurement()

        print("SFM3000 connected")

    def write_command(self, command):
        msb = (command >> 8) & 0xFF
        lsb = command & 0xFF

        msg = i2c_msg.write(I2C_ADDRESS, [msb, lsb])
        self.bus.i2c_rdwr(msg)

    def read_word(self):
        msg = i2c_msg.read(I2C_ADDRESS, 3)
        self.bus.i2c_rdwr(msg)

        data = list(msg)

        msb = data[0]
        lsb = data[1]
        crc = data[2]

        if crc8(bytes([msb, lsb])) != crc:
            raise RuntimeError("SFM3000 CRC error")

        return (msb << 8) | lsb

    def start_measurement(self):
        for i in range(5):
            try:
                self.write_command(0x1000)
                time.sleep(0.1)
                return
            except OSError as e:
                print(f"SFM3000 start failed {i+1}/5: {e}", flush=True)
                time.sleep(0.2)
        raise

    def step(self):
        raw = self.read_word()
        self.flow = (raw - OFFSET) / SCALE_FACTOR
        self.sample_monotonic = time.monotonic()

    def get_flow(self):
        return self.flow

    def get_sample(self):
        return self.flow, self.sample_monotonic


class AerosolFlowmeter:
    """Cached SDP8xx pressure, temperature, and calibrated aerosol flow."""

    def __init__(self, bus=1, address=0x25, calibration_lpm_per_pa=0.15228426395939088):
        # Keep this optional hardware dependency out of normal startup/imports.
        from sensirion_i2c_driver import I2cConnection, LinuxI2cTransceiver
        from sensirion_i2c_sdp import SdpI2cDevice

        transceiver = LinuxI2cTransceiver(f"/dev/i2c-{int(bus)}")
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
