import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path


gp_module = types.ModuleType("GP8XXX_IIC")
gp_module.GP8403 = object
sys.modules.setdefault("GP8XXX_IIC", gp_module)

serial_module = types.ModuleType("serial")
serial_module.Serial = object
serial_module.SEVENBITS = 7
serial_module.EIGHTBITS = 8
serial_module.PARITY_EVEN = "E"
serial_module.PARITY_NONE = "N"
serial_module.STOPBITS_ONE = 1
sys.modules.setdefault("serial", serial_module)

smbus_module = types.ModuleType("smbus2")
smbus_module.SMBus = object
smbus_module.i2c_msg = types.SimpleNamespace(write=lambda *args: None, read=lambda *args: None)
sys.modules.setdefault("smbus2", smbus_module)

gpio_module = types.ModuleType("gpiozero")
gpio_module.OutputDevice = object
gpio_module.PWMOutputDevice = object
sys.modules.setdefault("gpiozero", gpio_module)

spec = importlib.util.spec_from_file_location(
    "dmps_hardware", Path(__file__).parents[1] / "DmpsControl" / "hardware.py"
)
hardware = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hardware)


class FakeBus:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FlowmeterTests(unittest.TestCase):
    def test_sensirion_crc_vector(self):
        self.assertEqual(hardware.crc8(bytes([0xBE, 0xEF])), 0x13)

    def test_repeated_crc_failure_closes_and_reconnects_bus(self):
        old_bus = FakeBus()
        new_bus = FakeBus()
        meter = hardware.Flowmeter.__new__(hardware.Flowmeter)
        meter.bus_number = 1
        meter.address = 0x40
        meter.bus_factory = lambda number: new_bus
        meter.max_errors = 2
        meter.bus = old_bus
        meter.flow = 0.0
        meter.sample_monotonic = None
        meter.connected = True
        meter.error_count = 0
        meter.consecutive_errors = 1
        meter.crc_error_count = 0
        meter.reconnect_count = 0
        meter.last_error = None
        meter.serial_number = 123
        meter.article_number = 456
        meter.scale_factor = 140.0
        meter.offset = 32000.0
        meter._lock = threading.RLock()
        meter.read_word = lambda: (_ for _ in ()).throw(hardware.SFM3000CRCError("bad CRC"))
        meter.start_measurement = lambda: None

        with self.assertRaises(hardware.SFM3000CRCError):
            meter.step()

        self.assertTrue(old_bus.closed)
        self.assertIs(meter.bus, new_bus)
        self.assertTrue(meter.connected)
        self.assertEqual(meter.error_count, 1)
        self.assertEqual(meter.crc_error_count, 1)
        self.assertEqual(meter.reconnect_count, 1)


if __name__ == "__main__":
    unittest.main()
