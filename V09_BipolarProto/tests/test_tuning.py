import unittest
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "sheath_tuning", Path(__file__).parents[1] / "DmpsControl" / "tuning.py"
)
tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning)
aerosol_factor_from_pressures = tuning.aerosol_factor_from_pressures
lambda_pi_from_step = tuning.lambda_pi_from_step


def response_samples(final=10.0):
    rows = [
        {"phase": "low", "flow_lpm": 5.0, "step_elapsed_sec": 0.0}
        for _ in range(8)
    ]
    values = [5.0, 5.1, 5.5, 6.5, 8.2, final, final, final]
    rows.extend(
        {"phase": "high", "flow_lpm": value, "step_elapsed_sec": index + 1.0}
        for index, value in enumerate(values)
    )
    return rows


class TuningAnalysisTests(unittest.TestCase):
    def test_lambda_pi_is_conservative_and_positive(self):
        result = lambda_pi_from_step(response_samples(), 2.0, 3.0)
        self.assertGreater(result["Kp"], 0)
        self.assertGreater(result["Ki"], 0)
        self.assertEqual(result["Kd"], 0)
        self.assertGreaterEqual(result["lambda_sec"], 3 * result["tau_sec"])

    def test_rejects_no_response_and_reverse_gain(self):
        with self.assertRaisesRegex(ValueError, "No usable response"):
            lambda_pi_from_step(response_samples(final=5.1), 2.0, 3.0)
        with self.assertRaisesRegex(ValueError, "Reverse process gain"):
            lambda_pi_from_step(response_samples(final=3.0), 2.0, 3.0)

    def test_rejects_nonfinite_and_out_of_range(self):
        rows = response_samples()
        rows[2]["flow_lpm"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            lambda_pi_from_step(rows, 2.0, 3.0)
        rows = response_samples()
        rows[2]["flow_lpm"] = 101.0
        with self.assertRaisesRegex(ValueError, "outside"):
            lambda_pi_from_step(rows, 2.0, 3.0)

    def test_aerosol_factor_requires_stable_positive_pressure(self):
        result = aerosol_factor_from_pressures([2.0, 2.01, 1.99, 2.0, 2.0])
        self.assertAlmostEqual(result["factor_lpm_per_pa"], 0.5, places=3)
        with self.assertRaisesRegex(ValueError, "stable"):
            aerosol_factor_from_pressures([1.0, 1.2, 0.8, 1.1, 0.9])
        with self.assertRaisesRegex(ValueError, "positive"):
            aerosol_factor_from_pressures([-1.0] * 5)


if __name__ == "__main__":
    unittest.main()
