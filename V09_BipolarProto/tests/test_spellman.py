import ast
import threading
import unittest
from pathlib import Path


HV_PATH = Path(__file__).parents[1] / "DmpsControl" / "HV.py"


def load_spellman_class():
    module = ast.parse(HV_PATH.read_text())
    node = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "SpellmanHV"
    )
    namespace = {"threading": threading}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(HV_PATH), "exec"), namespace)
    return namespace["SpellmanHV"]


class SpellmanTests(unittest.TestCase):
    def make_device(self):
        device = load_spellman_class()()
        commands = []
        device.command = lambda command: commands.append(command)
        return device, commands

    def test_nonzero_voltage_reenables_after_disable(self):
        device, commands = self.make_device()
        device.enable()
        device.disable()
        device.set_voltage(1250)

        self.assertEqual(commands, ["0106EN=1", "0106EN=0", "0106EN=1", "0106V1=01250.0"])
        self.assertTrue(device.enabled)
        self.assertEqual(device.voltage, 1250)

    def test_zero_voltage_does_not_reenable_disabled_supply(self):
        device, commands = self.make_device()
        device.disable()
        device.zero()

        self.assertEqual(commands, ["0106EN=0", "0106V1=00000.0"])
        self.assertFalse(device.enabled)

    def test_enabled_supply_does_not_receive_redundant_enable(self):
        device, commands = self.make_device()
        device.enable()
        device.set_voltage(500)

        self.assertEqual(commands, ["0106EN=1", "0106V1=00500.0"])


if __name__ == "__main__":
    unittest.main()
