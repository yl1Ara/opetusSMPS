import numpy as np
from .ltubefl import ltubefl
from .cpc_loss import cpc_loss1, cpc_loss2
from .dmps_loss import dmps_loss1, dmps_loss2
from .teearra import teearra
from .varaus import varaus
from .gunn_woessner_modified import gunn_woessner_modified
from .calChargeFracF import calChargeFracF


def intfun(dp, t, press, p, volt, pituus, arkaksi, aryksi, qa, qc, qm, qs,
           pipelength, pipeflow, lsys, Zp, Zn, Mrp, Mrn, Np, Nn,
           charging_efficiency, summed, tube_segments=None, cpc_type=3010):
    """
    Main function for transfer function calculations.

    dp   : scalar or array, integration variable on log10 scale
    Returns a scalar when dp is scalar (as required by scipy.integrate.quad),
    or a 1-D array when dp is an array (for vectorised callers).
    """
    scalar_input = np.ndim(dp) == 0
    dp = np.atleast_1d(10.0 ** np.asarray(dp, dtype=float))

    # Laminar flow tube losses. Diameter and angle may be carried in
    # tube_segments for bookkeeping, but ltubefl currently uses length/flow.
    if tube_segments is None:
        tube_segments = (
            (np.nan, pipelength, pipeflow, 0.0),
            (np.nan, 2.80, 8 / 60000, 0.0),
            (np.nan, 5.21, 1.3 / 60000, 0.0),
        )

    tubeloss = np.ones_like(dp, dtype=float)
    for segment in tube_segments:
        _, length, flow, *_ = segment
        if np.isfinite(length) and np.isfinite(flow) and length > 0 and flow > 0:
            tubeloss *= ltubefl(dp, length, flow, t, press)
    

    if lsys == 1:
        cpcloss = cpc_loss1(dp, t, press, cpc_type=cpc_type)
    else:
        cpcloss = cpc_loss2(dp, t, press)

    if lsys == 1:
        dmaloss = dmps_loss1(dp, qa, t, press)
    else:
        dmaloss = dmps_loss2(dp, qa, t, press)

    totalloss = np.atleast_1d(tubeloss * cpcloss * dmaloss)

    # Transfer-function triangles; shape (n_dp, n_p)
    tr = teearra(p, dp, t, press, volt, pituus, arkaksi, aryksi, qa, qc, qm, qs)

    # Charging efficiency; shape (n_dp, n_p)
    p = np.atleast_1d(p)
    if np.any(~np.isfinite(p)) or np.any(p != np.rint(p)):
        raise ValueError("particle charge states must be finite integers")
    if charging_efficiency == 'wiedensohler':
        if summed == 1:
            charge = varaus(dp, -p, t) + varaus(dp, p, t)
        else:
            charge = varaus(dp, p, t)

    elif charging_efficiency == 'gunn woessner mod':
        charge = np.zeros((len(dp), len(p)))
        for i in range(len(p)):
            charge[:, i] = gunn_woessner_modified(
                p[i], dp, t, Zp, Zn, Mrp, Mrn, Np, Nn, summed)
            
    elif charging_efficiency == 'fuchs':
        charge = np.zeros((len(dp), len(p)))

        for i in range(len(dp)):
            frac, beta_p, beta_n = calChargeFracF(
                dp[i], Zp, Zn, Mrp, Mrn,
                Np=Np, Nn=Nn, epsp=1000, T=t, P=press
            )

            # The adaptive distribution is symmetric in index about q=0.
            center = len(frac) // 2
            idx = p.astype(int) + center

            if summed == 1:
                idx_opposite = (-p).astype(int) + center
                charge[i, :] = frac[idx] + frac[idx_opposite]
            else:
                charge[i, :] = frac[idx]

    else:
        raise ValueError(f"unknown charging efficiency: {charging_efficiency!r}")

    # Sum over charges, then multiply by losses; res shape (n_dp,)
    res = np.sum(tr * charge, axis=1) * totalloss

    if scalar_input:
        return float(res[0])
    return res
