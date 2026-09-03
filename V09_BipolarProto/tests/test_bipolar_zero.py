import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "zero_bipolar", Path(__file__).parents[1] / "deploy" / "zero-bipolar.py"
)
zero_bipolar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zero_bipolar)


class FakeHV:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def setup(self):
        self.calls.append("setup")

    def zero(self):
        self.calls.append("zero")
        if self.fail:
            raise RuntimeError("write failed")

    def cleanup(self):
        self.calls.append("cleanup")


class BipolarZeroTests(unittest.TestCase):
    def test_sets_zero_between_spi_setup_and_cleanup(self):
        hv = FakeHV()
        zero_bipolar.zero_bipolar(hv)
        self.assertEqual(hv.calls, ["setup", "zero", "cleanup"])

    def test_cleanup_runs_when_midpoint_write_fails(self):
        hv = FakeHV(fail=True)
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            zero_bipolar.zero_bipolar(hv)
        self.assertEqual(hv.calls, ["setup", "zero", "cleanup"])


if __name__ == "__main__":
    unittest.main()
