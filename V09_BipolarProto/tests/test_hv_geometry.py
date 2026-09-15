import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


HV_PATH = Path(__file__).parents[1] / "DmpsControl" / "HV.py"


def load_voltage_helpers():
    module = ast.parse(HV_PATH.read_text())
    nodes = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"cunningham_correction", "voltage_from_size"}
    ]
    namespace = {
        "np": np,
        "HaukeDMA": lambda length_m=0.28: SimpleNamespace(
            L=float(length_m), r1=0.025, r2=0.033,
        ),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HV_PATH), "exec"), namespace)
    return namespace


class HvGeometryTests(unittest.TestCase):
    def test_short_dma_requires_inverse_length_scaled_voltage(self):
        voltage_from_size = load_voltage_helpers()["voltage_from_size"]
        long_voltage = voltage_from_size(100, Q_sh_lpm=5, dma_length_m=0.28)
        short_voltage = voltage_from_size(100, Q_sh_lpm=5, dma_length_m=0.11)

        self.assertAlmostEqual(short_voltage / long_voltage, 0.28 / 0.11)

    def test_legacy_default_remains_28_cm(self):
        voltage_from_size = load_voltage_helpers()["voltage_from_size"]

        self.assertEqual(
            voltage_from_size(100, Q_sh_lpm=5),
            voltage_from_size(100, Q_sh_lpm=5, dma_length_m=0.28),
        )


if __name__ == "__main__":
    unittest.main()
