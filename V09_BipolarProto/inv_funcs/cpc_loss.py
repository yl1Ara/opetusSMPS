import numpy as np
from .ltubefl import ltubefl

def cpc_loss_curve(Dp, D50=None, D0=None, eta=None):
    if D50 is None:
        D50 = 6.2024e-9
        D0 = 4.6581e-9
        eta = 0.9041
    return (1-np.exp(-np.log(2)*((Dp-D0)/(D50-D0))))*eta

def cpc_loss1(dp, temp, press, cpc_type=3010):
    # Use one:
    """
    dp: float
    temp: float
    press: float
    """
    ione = 0
    
    if ione == 1:
        return np.ones(np.shape(dp))
    
    # Fit functions for CPC losses. Examples for old TSI cpcs exists.
    
    # The CPC type 3025,3022 or 3010
    
    # For TSI3010 you can give two elevated temperatures 25degC and 21degC
    TD = 21
    
    cpc_type = str(cpc_type)

    if cpc_type == "3772":
        # Ift calibration Td=29degC
        a = 98.68119
        b = 712.62942
        c = -3.5081
        d = 2.79325
        
        res = (a - b / (1.0 + np.exp((1e9 * dp - c) / d))) / 100
        
        # Handle both scalar and array inputs
        res = np.atleast_1d(res)
        iis = np.where(res <= 0)[0]
        if len(iis) > 0:
            res[iis] = 0
        if res.size == 1:
            res = res[0]
    
    elif cpc_type == "3022":
        # TSI3022
        X = dp / 0.01e-6
        res = 0.5 + 0.5 * (X - 1.0 / X) / (X + 1.0 / X)
    
    elif cpc_type == "3010":
        # TSI3010
        if TD == 25:
            a = 1.86
            D1 = 4.25
            D2 = 3.84
            DP50 = 5.7
        elif TD == 21:
            a = 1.4
            D1 = 6.5
            D2 = 1.9
            DP50 = 7.6
        else:
            a = 1.4
            D1 = 8.9
            D2 = 2.9
            DP50 = 10.5
        
        Dpp = 1e9 * dp
        D0 = D2 * np.log(a - 1) + D1
        
        res = 1 - a * (1 + np.exp((Dpp - D1) / D2)) ** (-1)
        iis = np.where(Dpp < D0)
        res[iis] = 0
    
    elif cpc_type == "3025":
        # TSI3025
        pipel = 0.1881
        pipef = 5.0 / 1e6
        
        DPCUT = 1.70e-9
        DPSLOPE = 4.75e-10
        EFFD1 = ltubefl(dp, pipel, pipef, temp, press)
        
        EFFD2 = 1.0 - np.exp((DPCUT - dp) / DPSLOPE)
        
        iis = np.where(dp <= DPCUT)
        EFFD2[iis] = 0
        
        res = EFFD1 * EFFD2
    
    elif cpc_type == "HY09":
        res = cpc_loss_curve(dp)
        res = np.where(dp > 20e-9, 1.0, res)
    else:
        # No validated counting-efficiency curve is available for this CPC.
        res = np.ones(np.shape(dp), dtype=float)
    
    return res


def cpc_loss2(dp, temp, press):
    # Stub for CPC loss2 - same as cpc_loss1 for now
    return cpc_loss1(dp, temp, press)
