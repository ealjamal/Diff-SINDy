import numpy as np
from findiff import Diff

def heat(t, x, u_i, p, source=0):
    dx = x[1] - x[0]
    d2 = Diff(0, dx, acc=4)**2
    u_xx = d2(u_i)

    heat_eq = p * u_xx + source

    return heat_eq

def burgers(t, x, u_i, p):
    dx = x[1] - x[0]
    d = Diff(0, dx, acc=4)
    d2 = d**2
    u_x = d(u_i)
    u_xx = d2(u_i)

    burgers_eq = -u_i * u_x + p * u_xx

    return burgers_eq

def parametric_cde(t, x, u_i, p):
    dx = x[1] - x[0]
    d = Diff(0, dx, acc=4)
    d2 = d**2
    u_x = d(u_i)
    u_xx = d2(u_i)

    parameteric_cde_eq = -(1 + p * np.sin(x)) * u_x + u_xx

    return parameteric_cde_eq

def wave(t, x, uv_i, p, damping=False):
    nx = len(x)
    u_i = uv_i[:nx]
    v_i = uv_i[nx:]
    
    dx = x[1] - x[0]
    d2_dx = Diff(0, dx, acc=4)**2
    u_xx = d2_dx(u_i)
    damping_term = -p * v_i if damping else 0

    wave_eq = p * u_xx + damping_term
    
    return np.concatenate([v_i, wave_eq])

def sine_gordan(t, x, uv_i, p):
    p1, p2 = p
    nx = len(x)
    u_i = uv_i[:nx]
    v_i = uv_i[nx:]

    dx = x[1] - x[0]
    d2_dx = Diff(0, dx, acc=4)**2
    u_xx = d2_dx(u_i)

    sine_gordan_eq = 4 * p1 * u_xx - p2 * np.sin(u_i)

    return np.concatenate([v_i, sine_gordan_eq])
    

