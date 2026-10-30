"""
One year of use, second by second, for a scooter controller in a city.

Use profiles (assumptions, stated in the report):
  commuter : Monday to Saturday, two rides a day, leaving 08:00 and 17:30
             local clock; each ride is WMTC part 1 three times
             (30 min, about 12 km).
  delivery : every day, twelve rides starting 11:00, 12:00, ..., 22:00;
             each ride is WMTC part 1 twice (20 min, about 8 km), so about
             96 km a day. Battery swaps between rides are assumed.

Thermal model of the controller:
  every MOSFET:   T_J = T_hs + P_fet (R_thJC + R_thCS)        (quasi-static)
  heatsink/case:  C_hs dT_hs/dt = P_total - (T_hs - T_amb) / R_hs
  P_total = 6 N_PAR P_fet (all six switches carry the same average loss)
R_hs and C_hs (assumptions): the controller box is an aluminium extrusion
of roughly 0.5 kg, so C_hs = 450 J/K (c_Al = 900 J/kg K); R_hs = 0.5 K/W
from box to air while riding. Both are varied in the sensitivity study.
T_amb is the NASA POWER 2 m air temperature at that hour, plus an optional
offset for heat inside the scooter body (0 K in the base case).
"""
import datetime as dt
import numpy as np

import wmtc
import vehicle
import mosfet
import weather

PROFILES = {
    "commuter": {"starts": [8.0, 17.5], "repeat": 3, "days": "Mon-Sat"},
    "delivery": {"starts": [float(h) for h in range(11, 23)], "repeat": 2, "days": "all"},
}
R_HS = 0.5
C_HS = 450.0
T_OFFSET = 0.0
DAYS = 365
JAN1_WEEKDAY = dt.date(2025, 1, 1).weekday()      # 2 = Wednesday


def ride_current(repeat, regen=0.0, i_lim=vehicle.I_PH_LIM, p_lim=vehicle.P_IN_LIM, m_extra=0.0):
    v_t = np.tile(wmtc.load("part1"), repeat)
    v, tq = vehicle.follow(v_t, i_lim=i_lim, p_lim=p_lim, regen=regen, m_extra=m_extra)
    return vehicle.phase_current_peak(tq), v


def day_schedule(profile, regen=0.0, i_lim=vehicle.I_PH_LIM, p_lim=vehicle.P_IN_LIM, m_extra=0.0):
    """Seconds from day start t0 (local hours) and the phase current over the day."""
    p = PROFILES[profile]
    i_ride, v_ride = ride_current(p["repeat"], regen, i_lim, p_lim, m_extra)
    t0 = p["starts"][0] - 0.25
    t1 = p["starts"][-1] + len(i_ride) / 3600.0 + 1.0
    n = int(round((t1 - t0) * 3600))
    i_day = np.zeros(n)
    km = 0.0
    for s in p["starts"]:
        k = int(round((s - t0) * 3600))
        i_day[k:k + len(i_ride)] = i_ride
        km += v_ride.sum() / 3600
    return t0, i_day, km


def riding_days(profile):
    wd = (JAN1_WEEKDAY + np.arange(DAYS)) % 7
    return np.ones(DAYS, bool) if PROFILES[profile]["days"] == "all" else wd != 6


def simulate(city, profile, n_par=mosfet.N_PAR, f_sw=mosfet.F_SW, sw_scale=mosfet.SW_SCALE,
             r_hs=R_HS, c_hs=C_HS, t_offset=T_OFFSET, rds_base=mosfet.RDS_TYP, regen=0.0,
             i_lim=vehicle.I_PH_LIM, p_lim=vehicle.P_IN_LIM, v_dc=vehicle.V_DC, m_extra=0.0):
    """Returns per-day arrays (days x seconds) of T_J and T_hs, and the
    ambient matrix. All 365 days are stepped together (vectorised)."""
    t_h, t2m = weather.city_temperature(city)
    t0, i_day, km = day_schedule(profile, regen, i_lim, p_lim, m_extra)
    ride = riding_days(profile)
    n = len(i_day)
    secs = np.arange(n) / 3600.0
    t_abs = (np.arange(DAYS)[:, None] * 24.0 + t0) + secs[None, :]
    t_amb = (weather.ambient_at(t_abs.ravel(), t_h, t2m).reshape(DAYS, n) + t_offset).astype(np.float64)
    r_jh = mosfet.R_JC + mosfet.R_CS
    th = t_amb[:, 0].copy()
    tj_prev = th.copy()
    TJ = np.empty((DAYS, n), np.float32)
    TH = np.empty((DAYS, n), np.float32)
    for k in range(n):
        i_k = i_day[k] * ride
        p_fet = mosfet.loss_per_fet(i_k, tj_prev, v_dc, n_par, f_sw, sw_scale, rds_base)
        p_tot = 6 * n_par * p_fet
        th = th + (p_tot - (th - t_amb[:, k]) / r_hs) / c_hs
        tj = th + p_fet * r_jh
        TJ[:, k] = tj
        TH[:, k] = th
        tj_prev = tj
    return {"tj": TJ, "th": TH, "t_amb": t_amb.astype(np.float32), "ride": ride,
            "km_per_day": km, "t0": t0, "i_day": i_day}
