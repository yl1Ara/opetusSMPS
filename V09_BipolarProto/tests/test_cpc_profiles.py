import importlib.util
import sys
import types
import unittest
from pathlib import Path


class FakeSerial:
    def __init__(self, **kwargs):
        self.settings = kwargs
        self.responses = []
        self.writes = []
        self.read_terminators = []
        self.is_open = True

    def reset_input_buffer(self):
        pass

    def write(self, payload):
        self.writes.append(payload)

    def flush(self):
        pass

    def read_until(self, terminator):
        self.read_terminators.append(terminator)
        return self.responses.pop(0) if self.responses else b""

    def close(self):
        self.is_open = False


gp_module = types.ModuleType("GP8XXX_IIC")
gp_module.GP8403 = object
sys.modules.setdefault("GP8XXX_IIC", gp_module)

serial_module = types.ModuleType("serial")
serial_module.Serial = FakeSerial
serial_module.SEVENBITS = 7
serial_module.EIGHTBITS = 8
serial_module.PARITY_EVEN = "E"
serial_module.PARITY_NONE = "N"
serial_module.STOPBITS_ONE = 1
sys.modules["serial"] = serial_module

smbus_module = types.ModuleType("smbus2")
smbus_module.SMBus = object
smbus_module.i2c_msg = types.SimpleNamespace(write=lambda *args: None, read=lambda *args: None)
sys.modules.setdefault("smbus2", smbus_module)

gpio_module = types.ModuleType("gpiozero")
gpio_module.OutputDevice = object
gpio_module.PWMOutputDevice = object
sys.modules.setdefault("gpiozero", gpio_module)

spec = importlib.util.spec_from_file_location(
    "dmps_hardware_cpc", Path(__file__).parents[1] / "DmpsControl" / "hardware.py"
)
hardware = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hardware)


class CPCProfileTests(unittest.TestCase):
    def test_uses_documented_serial_profile_and_cr_delimiter(self):
        cpc = hardware.CPC("/dev/test", "3010")
        cpc.ser.responses = [b"1234.5\r"]

        self.assertEqual(cpc.read_instrument(), "1234.5")
        self.assertEqual(cpc.ser.settings["baudrate"], 9600)
        self.assertEqual(cpc.ser.settings["bytesize"], 7)
        self.assertEqual(cpc.ser.settings["parity"], "E")
        self.assertEqual(cpc.ser.settings["stopbits"], 1)
        self.assertEqual(cpc.ser.writes, [b"RD\r"])
        self.assertEqual(cpc.ser.read_terminators, [b"\r"])

    def test_3771_uses_its_documented_serial_profile(self):
        cpc = hardware.CPC("/dev/test", "3771")

        self.assertEqual(cpc.type, "3771")
        self.assertEqual(cpc.concentration_command, "RD")
        self.assertEqual(cpc.ser.settings["baudrate"], 115200)
        self.assertEqual(cpc.ser.settings["bytesize"], 8)
        self.assertEqual(cpc.ser.settings["parity"], "N")
        self.assertEqual(cpc.ser.settings["stopbits"], 1)

    def test_skips_optional_echo_and_reports_protocol_error(self):
        cpc = hardware.CPC("/dev/test", "3010")
        cpc.ser.responses = [b"R5\r", b"READY\r"]
        self.assertEqual(cpc.query("R5"), "READY")

        cpc.ser.responses = [b"ERROR\r"]
        with self.assertRaisesRegex(RuntimeError, "CPC rejected R5"):
            cpc.query("R5")


if __name__ == "__main__":
    unittest.main()
