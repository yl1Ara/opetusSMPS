from GP8XXX_IIC import GP8403
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

    def get_flow(self):
        return self.flow


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
