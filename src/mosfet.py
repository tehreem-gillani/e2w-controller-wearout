"""
Power MOSFET and inverter loss model.

Device: Infineon IRFB4115PbF, 150 V, TO-220 (datasheet Rev. 2.1, 2026-08-03).
Values taken from the datasheet:
  R_DS(on)       9.3 mOhm typ, 11 mOhm max  (T_J = 25 C, V_GS = 10 V, I_D = 62 A)
  Fig. 4         normalised R_DS(on) vs T_J, read by eye at the points below
                 (reading error about +-0.05)
  t_r, t_f       73 ns, 39 ns   (V_DD = 98 V, I_D = 62 A, R_G = 2.2 Ohm)
  Q_rr           300 nC at 25 C, 450 nC at 125 C (V_DD = 130 V, I_F = 62 A, di/dt = 100 A/us)
  R_thJC         0.4 C/W max; Fig. 13 Foster network R = 0.245 / 0.155 C/W,
                 tau = 5.9149 / 0.6322 ms
  R_thCS         0.50 C/W typ (flat, greased)
  T_J max        175 C

Inverter (assumptions, varied in the sensitivity study):
  three-phase two-level bridge, N_PAR MOSFETs in parallel per switch
  (an "18-MOSFET controller" has N_PAR = 3), complementary gating
  (synchronous rectification), PWM frequency F_SW.

Losses averaged over one electrical period, per MOSFET:
  conduction : the phase current always flows through one switch of the
               leg, so each switch carries on average half of R I_rms^2:
               P_cond = R(T_J) (I_pk/N)^2 / 4
  switching  : hard switching, linear in current and voltage,
               E = 0.5 V_dc I (t_r + t_f), with the mean |i| over a period
               2 I_pk / pi, shared by the two switches of the leg, plus
               reverse recovery Q_rr V_dc per period of PWM.
  The switching times are the datasheet values at R_G = 2.2 Ohm; a real
  controller gate drive is usually slower, so SW_SCALE multiplies them.
The electrical period of a hub motor at walking speed and above is
milliseconds, much shorter than the thermal time constants of the
package and heatsink (seconds), so period-averaged losses are used.
The Foster time constants of R_thJC are below 6 ms, so at 1 s steps the
junction-to-case drop is quasi-static: T_J = T_C + P R_thJC.
"""
import numpy as np

RDS_TYP = 9.3e-3
RDS_MAX = 11e-3
# Fig. 4, normalised R_DS(on) (read by eye)
FIG4_T = np.array([-60, -40, -20, 0, 20, 25, 40, 60, 80, 100, 120, 140, 160, 175], float)
FIG4_R = np.array([0.57, 0.62, 0.70, 0.83, 0.97, 1.00, 1.12, 1.30, 1.52, 1.77, 2.03, 2.33, 2.65, 2.90])
T_R, T_F = 73e-9, 39e-9
QRR_25, QRR_125 = 300e-9, 450e-9
R_JC = 0.4
R_CS = 0.5
TJ_MAX = 175.0

N_PAR = 3
F_SW = 15e3
SW_SCALE = 2.0


def rds(tj, base=RDS_TYP):
    return base * np.interp(tj, FIG4_T, FIG4_R)


def qrr(tj):
    return np.interp(tj, [25.0, 125.0], [QRR_25, QRR_125])


def loss_per_fet(i_pk, tj, v_dc, n_par=N_PAR, f_sw=F_SW, sw_scale=SW_SCALE, base=RDS_TYP):
    """Average loss of one MOSFET (W) for peak phase current i_pk (A)."""
    i = np.asarray(i_pk, float) / n_par
    p_cond = rds(tj, base) * i * i / 4.0
    i_mean = 2.0 * i / np.pi
    e_sw = 0.5 * v_dc * i_mean * (T_R + T_F) * sw_scale
    p_sw = 0.5 * f_sw * e_sw                       # each switch of the leg hard-switches half the period
    p_rr = np.where(i > 0.1, 0.5 * f_sw * qrr(tj) * v_dc, 0.0)
    return p_cond + p_sw + p_rr
