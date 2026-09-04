"""Bipolar equilibrium charging with a Fuchs-type limiting-sphere method.

This is a corrected port of the project implementation attributed to Chen's
doctoral work. It has not been validated as a complete implementation of the
three-body trapping treatment in Hoppel and Frick (1986). Particle charge
``q`` is signed in elementary-charge units; positive and negative ions
therefore see opposite Coulomb potentials. All dimensional inputs and outputs
use SI units.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import logsumexp


NA = 6.02214076e23
kB = 1.380649e-23
Rg = NA * kB
Mr = 28.959
amu = 1.66053906660e-27
e = 1.602176634e-19
eps0 = 8.8541878128e-12


def cal_delta(a, lam):
    """Return the Fuchs limiting-sphere radius."""
    term1 = (1 + lam / a) ** 5 / 5
    term2 = (1 + lam**2 / a**2) * (1 + lam / a) ** 3 / 3
    term3 = 2 / 15 * (1 + lam**2 / a**2) ** 2.5
    return a**3 / lam**2 * (term1 - term2 + term3)


def cal_U(r, q, a, a_i=None, epsp=1000):
    """Coulomb plus image potential for a singly positive ion."""
    r = np.asarray(r, dtype=float)
    image_factor = (epsp - 1) / (epsp + 2)
    image_denominator = 2 * r**2 * (r**2 - a**2)
    return e**2 / (4 * np.pi * eps0) * (
        q / r - image_factor * a**3 / image_denominator
    )


def cal_b(r, delta, q, a, a_i=None, epsp=1000, T=293):
    """Return the impact parameter for trajectories reaching radius ``r``."""
    r = np.asarray(r, dtype=float)
    term = 1 + 2 / (3 * kB * T) * (
        cal_U(delta, q, a, a_i, epsp) - cal_U(r, q, a, a_i, epsp)
    )
    return np.where(term > 0, r * np.sqrt(np.maximum(term, 0)), 0.0)


def cal_bmin(delta, q, a, a_i=None, epsp=1000, T=293):
    """Numerically determine the limiting-sphere capture impact parameter."""
    r = np.linspace(a * (1 + 1e-9), delta, 1024)
    return float(np.min(cal_b(r, delta, q, a, a_i, epsp, T)))


def cal_beta(delta, q, a, a_i, D, c, alpha, epsp, T):
    """Return an ion-particle attachment coefficient in m3/s."""
    potential_delta = cal_U(delta, q, a, a_i, epsp)
    boltzmann_delta = np.exp(-potential_delta / (kB * T))
    numerator = np.pi * alpha * c * delta**2 * boltzmann_delta
    if numerator == 0:
        return 0.0

    x = np.linspace(np.finfo(float).eps, a / delta, 2048)
    integrand = np.exp(np.clip(cal_U(a / x, q, a, a_i, epsp) / (kB * T), -745, 700))
    potential_integral = trapezoid(integrand, x)
    denominator = 1 + (
        boltzmann_delta * alpha * c * delta**2 / (4 * D * a)
    ) * potential_integral
    return float(numerator / denominator)


def cal_char_frac(t, y, params):
    """Compatibility helper for the finite-state bipolar balance equations."""
    beta_p, beta_n, charges, Np, Nn = params
    dydt = np.zeros_like(y)
    for index in range(len(charges)):
        if index > 0:
            dydt[index] += beta_p[index - 1] * y[index - 1] * Np
        if index < len(charges) - 1:
            dydt[index] += beta_n[index + 1] * y[index + 1] * Nn
        if index < len(charges) - 1:
            dydt[index] -= beta_p[index] * Np * y[index]
        if index > 0:
            dydt[index] -= beta_n[index] * Nn * y[index]
    return dydt


def _validate_inputs(dp, Zp, Zn, Mrp, Mrn, Np, Nn, epsp, T, P):
    values = {
        "dp": dp, "Zp": Zp, "Zn": Zn, "Mrp": Mrp, "Mrn": Mrn,
        "Np": Np, "Nn": Nn, "T": T, "P": P,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(epsp) or epsp <= 1:
        raise ValueError("epsp must be finite and greater than one")


def _attachment_coefficients(dp, charges, Zp, Zn, Mrp, Mrn, epsp, T):
    ma = Mr * amu
    mp = Mrp * amu
    mn = Mrn * amu
    Dp = kB * T * Zp / e
    Dn = kB * T * Zn / e
    cp = np.sqrt(8 * kB * T / (np.pi * mp))
    cn = np.sqrt(8 * kB * T / (np.pi * mn))
    lambdap = 16 * np.sqrt(2) / (3 * np.pi) * Dp / cp * np.sqrt(ma / (ma + mp))
    lambdan = 16 * np.sqrt(2) / (3 * np.pi) * Dn / cn * np.sqrt(ma / (ma + mn))

    radius = dp / 2
    delta_p = cal_delta(radius, lambdap)
    delta_n = cal_delta(radius, lambdan)

    beta_p = np.empty(len(charges))
    beta_n = np.empty(len(charges))
    for index, charge in enumerate(charges):
        bmin_p = cal_bmin(delta_p, charge, radius, None, epsp, T)
        # A negative ion sees the Coulomb potential for the opposite charge.
        bmin_n = cal_bmin(delta_n, -charge, radius, None, epsp, T)
        beta_p[index] = cal_beta(
            delta_p, charge, radius, None, Dp, cp,
            (bmin_p / delta_p) ** 2, epsp, T,
        )
        beta_n[index] = cal_beta(
            delta_n, -charge, radius, None, Dn, cn,
            (bmin_n / delta_n) ** 2, epsp, T,
        )
    return beta_p, beta_n


def _steady_state(beta_p, beta_n, Np, Nn):
    center = len(beta_p) // 2
    log_weights = np.full(len(beta_p), -np.inf)
    log_weights[center] = 0.0

    for index in range(center + 1, len(beta_p)):
        up = beta_p[index - 1] * Np
        down = beta_n[index] * Nn
        if up <= 0 or down <= 0 or not np.isfinite(log_weights[index - 1]):
            break
        log_weights[index] = log_weights[index - 1] + np.log(up) - np.log(down)

    for index in range(center - 1, -1, -1):
        down = beta_n[index + 1] * Nn
        up = beta_p[index] * Np
        if down <= 0 or up <= 0 or not np.isfinite(log_weights[index + 1]):
            break
        log_weights[index] = log_weights[index + 1] + np.log(down) - np.log(up)

    fractions = np.exp(log_weights - logsumexp(log_weights))
    return fractions


@lru_cache(maxsize=4096)
def _cal_charge_fraction_cached(
    dp, Zp, Zn, Mrp, Mrn, Np, Nn, epsp, T, tail_tolerance, max_charge_limit,
):
    max_charge = 5
    while True:
        charges = np.arange(-max_charge, max_charge + 1, dtype=int)
        beta_p, beta_n = _attachment_coefficients(
            dp, charges, Zp, Zn, Mrp, Mrn, epsp, T,
        )
        fractions = _steady_state(beta_p, beta_n, Np, Nn)
        if fractions[0] + fractions[-1] <= tail_tolerance:
            return fractions, beta_p, beta_n
        if max_charge >= max_charge_limit:
            raise RuntimeError(
                "charge-state distribution did not converge before "
                f"|q|={max_charge_limit}"
            )
        max_charge = min(max_charge + 2, max_charge_limit)


def calChargeFracF(
    dp, Zp, Zn, Mrp, Mrn, Np=1e13, Nn=1e13, epsp=1000, T=293,
    P=101325, tail_tolerance=1e-10, max_charge_limit=31,
):
    """Return equilibrium fractions and attachment coefficients for ``-m..m``.

    The state-space half-width ``m`` starts at five and grows until the summed
    endpoint probability is below ``tail_tolerance``. Thus the returned array
    length can exceed eleven; charge zero is always at ``len(result) // 2``.
    """
    if not np.isfinite(tail_tolerance) or not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must be between zero and one")
    if int(max_charge_limit) != max_charge_limit or max_charge_limit < 5:
        raise ValueError("max_charge_limit must be an integer of at least five")
    _validate_inputs(dp, Zp, Zn, Mrp, Mrn, Np, Nn, epsp, T, P)
    result = _cal_charge_fraction_cached(
        float(dp), float(Zp), float(Zn), float(Mrp), float(Mrn), float(Np),
        float(Nn), float(epsp), float(T), float(tail_tolerance),
        int(max_charge_limit),
    )
    return tuple(values.copy() for values in result)


def fuchs_charge_fractions(dp, charges, **kwargs):
    """Return Fuchs fractions for requested signed particle charge states."""
    numeric = np.atleast_1d(np.asarray(charges, dtype=float))
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.rint(numeric)):
        raise ValueError("charges must contain finite integers")
    requested = numeric.astype(int)
    fractions, _, _ = calChargeFracF(dp, **kwargs)
    center = len(fractions) // 2
    indices = requested + center
    if np.any(indices < 0) or np.any(indices >= len(fractions)):
        raise ValueError("requested charge lies outside the converged state space")
    return fractions[indices]
