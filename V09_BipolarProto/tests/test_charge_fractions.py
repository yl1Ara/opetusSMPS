import unittest
from unittest.mock import patch

import numpy as np

from inv_funcs.calChargeFracF import calChargeFracF, fuchs_charge_fractions
from inv_funcs.gunn_woessner_modified import gunn_woessner_modified
from inv_funcs.intfun import intfun
from inv_funcs.varaus import varaus
from inv_funcs.wiedensohler_func import wiedensohler


class WiedensohlerReferenceTests(unittest.TestCase):
    """Published Wiedensohler (1988) coefficient reference points."""

    def test_reference_values_at_100_nm(self):
        diameter = 100e-9
        np.testing.assert_allclose(
            wiedensohler(diameter, "+"),
            [0.21379620895022336, 0.0317102743894808],
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            wiedensohler(diameter, "-"),
            [0.2793186922253972, 0.056078966337455474],
            rtol=2e-12,
        )

    def test_vectorized_and_scalar_implementations_agree(self):
        diameters = np.array([10, 20, 50, 100, 500], dtype=float) * 1e-9
        for polarity, charges in (("+", [1, 2]), ("-", [-1, -2])):
            expected = np.vstack([wiedensohler(dp, polarity) for dp in diameters])
            np.testing.assert_allclose(varaus(diameters, charges, 293.15), expected)


class FuchsChargeBalanceTests(unittest.TestCase):
    nominal = {
        "Zp": 1.35e-4,
        "Zn": 1.60e-4,
        "Mrp": 140,
        "Mrn": 101,
    }

    def test_distribution_is_normalized_with_small_tail(self):
        fractions, _, _ = calChargeFracF(500e-9, **self.nominal)
        self.assertAlmostEqual(float(fractions.sum()), 1.0, places=14)
        self.assertLessEqual(fractions[0] + fractions[-1], 1e-10)
        self.assertGreater(len(fractions), 11)

    def test_equal_ions_give_charge_sign_symmetry(self):
        fractions, beta_p, beta_n = calChargeFracF(
            100e-9, 1.5e-4, 1.5e-4, 120, 120,
        )
        np.testing.assert_allclose(fractions, fractions[::-1], rtol=2e-12, atol=1e-15)
        np.testing.assert_allclose(beta_p, beta_n[::-1], rtol=2e-12, atol=0)

    def test_nominal_faster_negative_ions_favor_negative_particles(self):
        negative, positive = fuchs_charge_fractions(
            40e-9, [-1, 1], **self.nominal,
        )
        self.assertGreater(negative, positive)

    def test_stationary_detailed_balance(self):
        fractions, beta_p, beta_n = calChargeFracF(100e-9, **self.nominal)
        left = fractions[:-1] * beta_p[:-1] * 1e13
        right = fractions[1:] * beta_n[1:] * 1e13
        active = (left > 0) | (right > 0)
        np.testing.assert_allclose(left[active], right[active], rtol=2e-12, atol=0)

    def test_invalid_physical_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dp"):
            calChargeFracF(0, **self.nominal)

    def test_nonintegral_charge_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite integers"):
            fuchs_charge_fractions(40e-9, [1.5], **self.nominal)


class GunnWoessnerConventionTests(unittest.TestCase):
    def test_zn_over_zp_above_one_favors_negative_charge(self):
        common = (np.array([100e-9]), 293.15, 1e-4, 1.2e-4, 140, 101, 1e13, 1e13)
        positive = gunn_woessner_modified(1, *common, 0)[0]
        negative = gunn_woessner_modified(-1, *common, 0)[0]
        self.assertGreater(negative, positive)


class InversionChargeMatrixTests(unittest.TestCase):
    def _response_for_charge(self, charge):
        log_diameter = np.log10(np.array([40e-9]))
        with (
            patch("inv_funcs.intfun.teearra", return_value=np.ones((1, 1))),
            patch("inv_funcs.intfun.cpc_loss1", return_value=np.ones(1)),
            patch("inv_funcs.intfun.dmps_loss1", return_value=np.ones(1)),
        ):
            return intfun(
                log_diameter, 293.15, 101325, np.array([charge]), 1000,
                0.28, 0.033, 0.025, 1 / 60000, 10 / 60000,
                10 / 60000, 1 / 60000, 0, 1 / 60000, 1,
                1.35e-4, 1.60e-4, 140, 101, 1e13, 1e13,
                "fuchs", 0, tube_segments=(),
            )[0]

    def test_fuchs_matrix_uses_signed_particle_charge(self):
        self.assertGreater(self._response_for_charge(-1), self._response_for_charge(1))

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown charging efficiency"):
            self._response_with_unknown_model()

    def _response_with_unknown_model(self):
        with (
            patch("inv_funcs.intfun.teearra", return_value=np.ones((1, 1))),
            patch("inv_funcs.intfun.cpc_loss1", return_value=np.ones(1)),
            patch("inv_funcs.intfun.dmps_loss1", return_value=np.ones(1)),
        ):
            return intfun(
                np.log10(np.array([40e-9])), 293.15, 101325, np.array([-1]), 1000,
                0.28, 0.033, 0.025, 1 / 60000, 10 / 60000,
                10 / 60000, 1 / 60000, 0, 1 / 60000, 1,
                1.35e-4, 1.60e-4, 140, 101, 1e13, 1e13,
                "not-a-model", 0, tube_segments=(),
            )


if __name__ == "__main__":
    unittest.main()
