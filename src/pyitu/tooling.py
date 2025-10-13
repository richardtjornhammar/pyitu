import numpy as np

c0 = 299792458.0

def eps_water_double_debye_0(f_GHz, T_C=15.0):
    f_Hz = f_GHz*1e9
    eps_s, eps_1, eps_inf = 78.3, 5.2, 4.9
    tau1, tau2 = 8.27e-12, 0.10e-12
    w = 2*pi*f_Hz
    term1 = (eps_s - eps_1) / (1.0 - 1j*w*tau1)
    term2 = (eps_1 - eps_inf) / (1.0 - 1j*w*tau2)
    return eps_inf + term1 + term2

def water_permittivity_double_debye_0(f_GHz: float, T_C: float = 20.0) -> complex:
    # T_C for temperature dependance around dielectric standard at 20^oC
    eps_s = 78.3
    eps_1 = 5.2
    eps_inf = 4.9
    tau1 = 8.27e-12
    tau2 = 0.10e-12
    T_ref = 20.0
    dT = T_C - T_ref
    eps_s *= (1.0 - 0.002 * dT)
    tau1 *= (1.0 - 0.02 * dT/10.0)
    w = 2*pi*f_GHz*1e9
    term1 = (eps_s - eps_1) / (1.0 - 1j*w*tau1)
    term2 = (eps_1 - eps_inf) / (1.0 - 1j*w*tau2)
    return eps_inf + term1 + term2

def marshall_palmer_DSD(D_mm, R_mmph):
    if R_mmph <= 0:
        return np.zeros_like(D_mm)
    N0 = 8000.0
    Lambda = 4.1 * (R_mmph**-0.21)
    return N0 * np.exp(-Lambda * D_mm)

def fuzzy_find(v,arr) :
    idx = sorted( [(f,i) for (i,f) in zip( range(len(arr)),np.abs(arr-v) ) ] ,key = lambda x:x[0] )[0][1]
    return idx
