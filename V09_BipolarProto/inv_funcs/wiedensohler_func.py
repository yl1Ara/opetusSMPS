import numpy as np

# Wiedensohler (1988) polynomial coefficients.
# Row 0 = |q|=1, row 1 = |q|=2.
# From MATLAB wiedensohler.m (Runlong Cai's Julia code, 14th August 2023).
_COEFF = {
    '+': np.array([
        [-2.3484,  0.6044,  0.4800,  0.0013, -0.1553,  0.0320],
        [-44.4756, 79.3772, -62.89,  26.4492, -5.748,   0.5049],
    ]),
    '-': np.array([
        [-2.3197,  0.6175,  0.6201, -0.1105, -0.1260,  0.0297],
        [-26.3328, 35.9044, -21.4608,  7.0867, -1.3088,  0.1051],
    ]),
}


def wiedensohler(dp, polarity):
    """
    Wiedensohler (1988) charging fraction for |q|=1 and |q|=2.

    dp       : scalar particle diameter [m]
    polarity : '+' or '-'
    Returns  : ndarray shape (2,) — [f_q1, f_q2]
    """
    a = _COEFF[polarity]
    ldp = np.log10(float(dp) * 1e9)
    powers = ldp ** np.arange(6)            # [ldp^0, ldp^1, ..., ldp^5]
    return 10.0 ** (a @ powers)             # shape (2,)
