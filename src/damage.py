"""
From junction temperature to wear-out.

Cycle counting: rainflow (rainflow.py) on each day's T_J trace, 0.5 K
hysteresis. Each day starts and ends with the controller at air
temperature, so days are counted separately.

Life model: there is no validated power-cycling model for discrete TO-220
MOSFETs (review: W. Wang et al., Chinese J. Electr. Eng., 2025). We therefore use the common
Coffin-Manson-Arrhenius form only to RANK conditions, not to predict years:
    N_f(dT, T_m) = A dT^-n exp(Ea / (k_B T_m))
and anchor A to the one measured point we have for this exact part:
the Infineon product qualification report for IRFB4115 lists the
Intermittent Operational Life test (MIL-STD-750 method 1037) at
dT = 100 K for 8572 cycles with 0 failures out of 231 parts. Taking
N_f(100 K) = 8572 treats the test length as the life, which can only
overstate damage, so the absolute index below is pessimistic. The mean
temperature of that test is not in the report; we assume 75 C
(25 C to 125 C).
Base exponents: n = 5 (the LESIT module model) and Ea = 0.617 eV
(9.89e-20 J, the Coffin-Manson-Arrhenius model of Lan et al.), both as
listed in the review; n is varied 3-6 and Ea 0-0.617
eV. Cycles below DT_MIN (2 K) are left out; varied 0-5 K.

Outputs per year:
  D_index : sum of cycles / N_f; 1.0 means as much thermal cycling as the
            whole qualification test (not "end of life")
  ratio   : D_index of a city / D_index of Madrid with the same settings,
            the main result, which does not depend on A.
"""
import numpy as np
import rainflow

K_B = 8.617333e-5          # eV/K
N_ANCHOR, DT_ANCHOR, TM_ANCHOR = 8572.0, 100.0, 75.0
HYST = 0.5
DT_MIN = 2.0


def day_cycles(tj_day, hysteresis=HYST):
    return rainflow.count(tj_day, hysteresis)


def all_cycles(tj, ride_mask, hysteresis=HYST):
    """Concatenated cycles (range, mean, count) over all riding days."""
    out = []
    for d in np.flatnonzero(ride_mask):
        out.extend(day_cycles(tj[d], hysteresis))
    a = np.array(out, float) if out else np.zeros((0, 3))
    return a


def ride_windows(i_day, tail=900):
    """(start, end) index of each ride, from the first to the last second
    with current, plus `tail` seconds of cooling."""
    on = np.flatnonzero(i_day > 0)
    if len(on) == 0:
        return []
    gaps = np.flatnonzero(np.diff(on) > 300)        # stops inside a ride are shorter than 5 min
    starts = np.concatenate([[on[0]], on[gaps + 1]])
    ends = np.concatenate([on[gaps], [on[-1]]])
    return [(a, min(b + tail, len(i_day))) for a, b in zip(starts, ends)]


def active_cycles(tj, ride_mask, i_day, hysteresis=HYST):
    """Rainflow on each ride window separately. This keeps the cycles caused
    by riding (power cycling) and drops the slow day-long swing of the air
    temperature between a morning and an evening ride."""
    out = []
    win = ride_windows(i_day)
    for d in np.flatnonzero(ride_mask):
        for a, b in win:
            out.extend(rainflow.count(tj[d, a:b], hysteresis))
    return np.array(out, float) if out else np.zeros((0, 3))


def damage(cycles, n=5.0, ea=0.617, dt_min=DT_MIN):
    if len(cycles) == 0:
        return 0.0
    r, m, c = cycles[:, 0], cycles[:, 1], cycles[:, 2]
    use = r >= max(dt_min, 1e-6)
    r, m, c = r[use], m[use], c[use]
    a = N_ANCHOR * DT_ANCHOR ** n / np.exp(ea / (K_B * (TM_ANCHOR + 273.15)))
    nf = a * r ** (-n) * np.exp(ea / (K_B * (m + 273.15)))
    return float(np.sum(c / nf))


def stress_metrics(sim):
    tj, th, ride = sim["tj"], sim["th"], sim["ride"]
    moving = sim["i_day"] > 0
    tj_r = tj[ride][:, moving]
    th_r = th[ride]
    rise = (tj - sim["t_amb"])[ride][:, moving]
    return {"tj_max": float(tj.max()),
            "tj_p99_riding": float(np.percentile(tj_r, 99)),
            "tj_mean_riding": float(tj_r.mean()),
            "rise_mean_riding": float(rise.mean()),
            "riding_hours_tj_above_100": float((tj_r > 100).sum() / 3600),
            "th_max": float(th_r.max()),
            "riding_hours_th_above_80": float((th[ride][:, moving] > 80).sum() / 3600)}
