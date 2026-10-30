"""
Scooter longitudinal model and the motor phase current it needs.

Vehicle (the class of scooter sold in Pakistan and South Asia at this power):
  dry mass 102 kg, 2000 W hub motor, 70 km/h, 90/90-10 tyres
  (Evee C1 Air listing, gari.pk, checked 24 Oct 2026). Battery 72 V
  (PakWheels listing of the same scooter).
  Driver 75 kg (UN GTR No. 2, para 3.4), so mass in running order
  m_ref = 177 kg.

Road load: UN GTR No. 2, Annex 3, running resistance table for
  175 < m_ref <= 185 kg:  m_i = 180 kg,  a = 15.8 N,  b = 0.0227 N/(km/h)^2,
  F_road = a + b v^2  (v in km/h). The table was made for conventional
  motorcycles; we use it as the official reference road load.
Rotating parts: GTR No. 2 estimates the rotating mass of the front wheel as
  3 % of m (para 6.5.6.1.2.1) and the equivalent inertia mass of the rear
  wheel and the parts rotating with it (here the hub motor) as 4 % of m
  (para 6.5.6.1.2.2); we add both, 7 %.
Wheel radius: 90/90-10 -> 10 in rim + 2 x 0.9 x 90 mm = 416 mm diameter.

Motor (direct-drive hub PMSM, field-oriented control):
  Torque T = 1.5 * k_e * i_q with k_e the peak phase back-EMF per mech rad/s.
  k_e is set from the supply: the phase back-EMF at top speed is taken as
  90 % of V_dc / sqrt(3) (space-vector limit), an assumption. With
  V_dc = 72 V and 70 km/h this gives k_e = 0.40 V s/rad.
  Only i_q is used (no field weakening; part 1 of WMTC stays below 60 km/h).
  Regenerative braking: none in the base case (assumption; many low-cost
  controllers have no or weak regen). Sensitivity in stage 3.
"""
import numpy as np

M_DRY = 102.0
M_RIDER = 75.0
M_REF = M_DRY + M_RIDER
A_ROLL = 15.8            # N
B_AERO = 0.0227          # N/(km/h)^2
ROT = 1.07
M_EFF = M_REF * ROT
R_WHEEL = 0.416 / 2
V_DC = 72.0
V_MAX_KMH = 70.0
KE = 0.9 * (V_DC / np.sqrt(3)) / ((V_MAX_KMH / 3.6) / R_WHEEL)     # V s/rad (peak, per phase)
P_RATED = 2000.0


def speed_ms(v_kmh):
    return np.asarray(v_kmh, float) / 3.6


def wheel_torque(v_kmh, dt=1.0, regen=0.0):
    """Wheel (= motor) torque for a speed trace given at 1 s steps.
    Acceleration by central difference. Negative traction torque (braking)
    is supplied by the motor only up to the fraction `regen`; the rest goes
    to the mechanical brakes."""
    v = speed_ms(v_kmh)
    acc = np.gradient(v, dt)
    f_road = np.where(v > 0.01, A_ROLL + B_AERO * (v * 3.6) ** 2, 0.0)
    f = M_EFF * acc + f_road
    f = np.where(f < 0, regen * f, f)
    return f * R_WHEEL


def phase_current_peak(torque):
    """Peak phase current (A) for the torque, i = T / (1.5 k_e)."""
    return np.abs(torque) / (1.5 * KE)


def mech_power(v_kmh, torque):
    return torque * speed_ms(v_kmh) / R_WHEEL


# Controller limits (assumptions, varied in the sensitivity study). WMTC was
# written for motorcycles and asks for up to ~4 kW in part 1, more than a
# 2 kW scooter can give, so the scooter follows the trace as closely as its
# limits allow.
I_PH_LIM = 100.0          # A, peak phase current limit
P_IN_LIM = 3000.0         # W, battery-side power limit (about 42 A at 72 V)


def road_load(m_extra=0.0):
    """GTR No. 2 Annex 3 table formula: m_i is m_ref rounded to 10 kg,
    a = 0.088 m_i (N), b = 0.000015 m_i + 0.02 (N/(km/h)^2). For the base
    scooter (m_ref = 177 kg) this gives the table row a = 15.8, b = 0.0227."""
    mi = 10.0 * np.round((M_REF + m_extra) / 10.0)
    return round(0.088 * mi, 1), round(0.000015 * mi + 0.02, 4), (M_REF + m_extra) * ROT


def follow(v_target_kmh, i_lim=I_PH_LIM, p_lim=P_IN_LIM, eta=0.88, dt=1.0, regen=0.0, m_extra=0.0):
    """Drive the target trace with torque limits. Each second the rider asks
    for the force that reaches the next target speed; the controller gives
    at most min(i_lim current, eta * p_lim power), eta = 0.88 being an
    assumed motor + controller efficiency. Returns actual speed (km/h)
    and motor torque (N m)."""
    a_roll, b_aero, m_eff = road_load(m_extra)
    vt = speed_ms(v_target_kmh)
    n = len(vt)
    v = np.zeros(n)
    tq = np.zeros(n)
    t_i = 1.5 * KE * i_lim
    for k in range(n - 1):
        f_road = a_roll + b_aero * (v[k] * 3.6) ** 2 if v[k] > 0.01 else 0.0
        f_need = m_eff * (vt[k + 1] - v[k]) / dt + f_road
        t_need = f_need * R_WHEEL
        w = max(v[k], 0.5) / R_WHEEL
        t_max = min(t_i, eta * p_lim / w)
        if t_need >= 0:
            t = min(t_need, t_max)
            f_brake = 0.0
        else:
            t = max(regen * t_need, -t_max)
            f_brake = t_need / R_WHEEL - t / R_WHEEL      # mechanical brakes take the rest
        tq[k] = t
        v[k + 1] = max(v[k] + (t / R_WHEEL + f_brake - f_road) / m_eff * dt, 0.0)
        if vt[k + 1] == 0.0 and v[k + 1] < 0.3:
            v[k + 1] = 0.0
    return v * 3.6, tq
