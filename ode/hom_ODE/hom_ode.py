# hom_ode.py
import math
import numpy as np
from scipy.integrate import solve_ivp

def ode_rhs(t, y):
    """
    Homogeneous RD axion equation in cosmic time units t = M_phi * (cosmic time):
      theta'' + (3/(2 t)) theta' + sin(theta) = 0
    """
    th, dth = y
    ddth = -(3.0 / (2.0 * t)) * dth - math.sin(th)
    return [dth, ddth]

def energy_density(theta, dtheta):
    """
    E = 0.5*theta'^2 + (1 - cos(theta))
    """
    theta = np.asarray(theta, dtype=np.float64)
    dtheta = np.asarray(dtheta, dtype=np.float64)
    return 0.5 * dtheta**2 + (1.0 - np.cos(theta))

def Ea3_of_solution(t_arr, theta_arr, dtheta_arr):
    """
    For RD: a ∝ t^{1/2}  => a^3 ∝ t^{3/2}
    So conserved quantity in oscillatory regime:
      Ea3 = E(t) * t^{3/2}
    """
    E = energy_density(theta_arr, dtheta_arr)
    return E * (t_arr**1.5)

def late_time_Ea3(t_arr, theta_arr, dtheta_arr, late_frac=0.25, mode="time_weighted"):
    """
    Robust estimator for late-time plateau of Ea^3.

    mode:
      - "median": extremely robust, ignores non-uniform solver output density
      - "time_weighted": best physical averaging on irregular grids
    """
    n = len(t_arr)
    n_lat = max(int(n * late_frac), 20)
    sl = slice(n - n_lat, n)

    t = np.asarray(t_arr[sl], dtype=np.float64)
    ea3 = Ea3_of_solution(t, theta_arr[sl], dtheta_arr[sl])

    if mode == "median":
        return float(np.median(ea3))

    # time-weighted average on irregular grid
    if len(t) < 3:
        return float(np.mean(ea3))
    dt = np.diff(t)
    ea3_mid = 0.5 * (ea3[:-1] + ea3[1:])
    denom = np.sum(dt)
    if denom <= 0:
        return float(np.mean(ea3))
    return float(np.sum(ea3_mid * dt) / denom)

def _max_step(theta0, t0, t1):
    """
    Smaller step near hilltop for stability.
    """
    span = (t1 - t0)
    if span <= 0:
        return 1e-3
    # x ~ 0 near hilltop, x ~ O(1) far from hilltop
    x = abs((math.pi - abs(float(theta0))) / math.pi)
    # shrink max_step as x -> 0
    f = 0.05 + 0.95 * x
    return span / (3000.0 / f)

def solve_noPT(theta0, t_start=1e-3, t_end=800.0,
               method="DOP853", rtol=1e-9, atol=1e-11,
               late_frac=0.25, late_mode="time_weighted"):
    """
    Reference (no PT): mass on from early time.
    IC at t_start: theta = theta0, theta' = 0
    Returns: (Ea3_late, sol)
    """
    max_step = _max_step(theta0, t_start, t_end)

    sol = solve_ivp(
        ode_rhs,
        t_span=(t_start, t_end),
        y0=[float(theta0), 0.0],
        method=method,
        rtol=rtol, atol=atol,
        dense_output=False,
        max_step=max_step,
    )
    if not sol.success:
        return np.nan, sol

    Ea3 = late_time_Ea3(sol.t, sol.y[0], sol.y[1], late_frac=late_frac, mode=late_mode)
    return Ea3, sol

def solve_PT(theta0, t_p, t_end_min=800.0, extra_after=400.0,
             method="DOP853", rtol=1e-9, atol=1e-11,
             late_frac=0.25, late_mode="time_weighted"):
    """
    PT: field frozen until t_p; at t_p set IC (theta0, 0) and evolve.
    Returns: (Ea3_late, sol)
    """
    if (not np.isfinite(t_p)) or (t_p <= 0):
        return np.nan, None

    t_end = max(t_end_min, float(t_p) + float(extra_after))
    max_step = _max_step(theta0, t_p, t_end)

    sol = solve_ivp(
        ode_rhs,
        t_span=(float(t_p), float(t_end)),
        y0=[float(theta0), 0.0],
        method=method,
        rtol=rtol, atol=atol,
        dense_output=False,
        max_step=max_step,
    )
    if not sol.success:
        return np.nan, sol

    Ea3 = late_time_Ea3(sol.t, sol.y[0], sol.y[1], late_frac=late_frac, mode=late_mode)
    return Ea3, sol

def first_minimum_theta(sol, require_cross_zero=True):
    """
    Robust first minimum finder:
      looks for first turning point where theta' goes (-) -> (+),
      optionally requiring theta to have crossed below 0 before accepting.

    Returns theta_min (float) or np.nan.
    """
    if sol is None:
        return np.nan
    t = sol.t
    th = sol.y[0]
    dth = sol.y[1]

    crossed = (not require_cross_zero)

    for i in range(1, len(t)):
        if require_cross_zero and (not crossed) and (th[i] < 0.0):
            crossed = True

        if not crossed:
            continue

        if (dth[i-1] < 0.0) and (dth[i] > 0.0):
            # choose smaller theta in bracket as a stable estimate
            return float(min(th[i-1], th[i]))

    return np.nan
