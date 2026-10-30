"""
Runs the whole study and writes results/ and figures/.
    cd src && python run_all.py        (a few minutes)

Study:
  A  base case: 5 cities x 2 use profiles
  B  are the city rankings robust to the life-model choice? (n, Ea, dT_min)
  C  one-factor-at-a-time: which design / use factor matters most?
  D  a "hard use" case combining realistic worst factors, all cities
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import weather
import mission
import damage
import vehicle
import mosfet

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
CITIES = list(weather.CITIES)
PROFILES = ["commuter", "delivery"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.frameon": False, "savefig.bbox": "tight"})

BASE = {}
HARD = {"r_hs": 1.5, "n_par": 2, "t_offset": 8.0, "m_extra": 75.0}
FACTORS = [
    ("box to air R_hs 0.5 -> 1.5 K/W (no airflow)", {"r_hs": 1.5}),
    ("12 instead of 18 MOSFETs", {"n_par": 2}),
    ("+8 K inside the scooter body", {"t_offset": 8.0}),
    ("pillion / cargo +75 kg", {"m_extra": 75.0}),
    ("slower gate drive (x4 datasheet times)", {"sw_scale": 4.0}),
    ("R_DS(on) at datasheet max", {"rds_base": mosfet.RDS_MAX}),
    ("higher current limit 150 A, 4.5 kW", {"i_lim": 150.0, "p_lim": 4500.0}),
    ("regen braking (50 %)", {"regen": 0.5}),
]
LIFE_MODELS = [(n, ea, dtm) for n in (3.0, 4.0, 5.0, 6.0) for ea in (0.0, 0.3, 0.617) for dtm in (0.0, 2.0, 5.0)]


def run_case(city, profile, kw):
    s = mission.simulate(city, profile, **kw)
    cyc = damage.all_cycles(s["tj"], s["ride"])
    m = damage.stress_metrics(s)
    m["D_index_base"] = damage.damage(cyc)
    act = damage.active_cycles(s["tj"], s["ride"], s["i_day"])
    m["D_active_base"] = damage.damage(act)
    for n in (3.0, 4.0, 6.0):
        m[f"D_index_n{int(n)}"] = damage.damage(cyc, n=n)
    m["km_per_day"] = s["km_per_day"]
    m["n_cycles_ge2K"] = int(np.sum(cyc[:, 2][cyc[:, 0] >= 2.0])) if len(cyc) else 0
    s["active_cycles"] = act
    return s, cyc, m


def main():
    out = {"weather": weather.summary(), "assumptions": {
        "vehicle": {"m_ref_kg": vehicle.M_REF, "ke_Vs_per_rad": vehicle.KE, "i_lim_A": vehicle.I_PH_LIM,
                    "p_lim_W": vehicle.P_IN_LIM, "v_dc": vehicle.V_DC},
        "controller": {"n_par": mosfet.N_PAR, "f_sw": mosfet.F_SW, "sw_scale": mosfet.SW_SCALE,
                       "r_hs": mission.R_HS, "c_hs": mission.C_HS},
        "life_model": {"anchor": [damage.N_ANCHOR, damage.DT_ANCHOR, damage.TM_ANCHOR],
                       "n": 5.0, "Ea_eV": 0.617, "dT_min": damage.DT_MIN, "hysteresis": damage.HYST}}}

    # ---------------- A + B
    A, cycles, act_cycles = {}, {}, {}
    sims = {}
    for p in PROFILES:
        for c in CITIES:
            s, cyc, m = run_case(c, p, BASE)
            A[f"{p}/{c}"] = m
            cycles[(p, c)] = cyc
            act_cycles[(p, c)] = s["active_cycles"]
            if p == "delivery" and c in ("Delhi", "Madrid", "Jakarta"):
                sims[c] = s
            print("A", p, c, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
    out["A_base"] = A
    B = {}
    for p in PROFILES:
        rows = []
        for (n, ea, dtm) in LIFE_MODELS:
            d = {c: damage.damage(cycles[(p, c)], n, ea, dtm) for c in CITIES}
            ratio = {c: d[c] / d["Madrid"] for c in CITIES}
            order = sorted(CITIES, key=lambda c: -d[c])
            da = {c: damage.damage(act_cycles[(p, c)], n, ea, dtm) for c in CITIES}
            ratio_a = {c: da[c] / da["Madrid"] for c in CITIES}
            order_a = sorted(CITIES, key=lambda c: -da[c])
            rows.append({"n": n, "Ea": ea, "dT_min": dtm, "ratio_to_Madrid": ratio, "order": order,
                         "active_ratio_to_Madrid": ratio_a, "active_order": order_a})
        B[p] = rows
    out["B_life_model"] = B

    # ---------------- C
    C = {}
    for city in ("Delhi", "Karachi", "Madrid"):
        base = A[f"delivery/{city}"]
        rows = [{"factor": "base", **{k: base[k] for k in ("D_index_base", "tj_max", "th_max", "riding_hours_th_above_80")}}]
        for name, kw in FACTORS:
            _, _, m = run_case(city, "delivery", kw)
            rows.append({"factor": name, **{k: m[k] for k in ("D_index_base", "tj_max", "th_max", "riding_hours_th_above_80")}})
            print("C", city, name, round(m["D_index_base"] / base["D_index_base"], 2), round(m["tj_max"], 1))
        C[city] = rows
    out["C_factors_delivery"] = C

    # ---------------- D
    D, hard_cycles = {}, {}
    for p in PROFILES:
        for c in CITIES:
            s, cyc, m = run_case(c, p, HARD)
            D[f"{p}/{c}"] = m
            hard_cycles[(p, c)] = cyc
            if p == "delivery" and c == "Delhi":
                sims["Delhi_hard"] = s
            print("D", p, c, round(m["D_index_base"], 6), round(m["tj_max"], 1), round(m["th_max"], 1))
    out["D_hard"] = {"settings": HARD, "results": D}
    out["D_hard_ratio_to_Madrid"] = {p: {c: D[f"{p}/{c}"]["D_index_base"] / D[f"{p}/Madrid"]["D_index_base"]
                                         for c in CITIES} for p in PROFILES}

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    figures(out, sims, cycles)
    write_summary(out)


def figures(out, sims, cycles):
    col = {"Karachi": "C0", "Delhi": "C3", "Jakarta": "C2", "Lagos": "C1", "Madrid": "C4"}
    # 1 ambient distributions
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for c in CITIES:
        _, T = weather.city_temperature(c)
        ax.plot(np.sort(T), np.linspace(0, 100, len(T)), color=col[c], label=c)
    ax.set_xlabel("air temperature 2025 (°C)"); ax.set_ylabel("share of hours below (%)"); ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "f1_ambient.png"), dpi=200); plt.close(fig)

    # 2 one day of junction temperature, delivery, Delhi hottest day, base and hard use
    s, h = sims["Delhi"], sims["Delhi_hard"]
    d = int(np.argmax(s["t_amb"].max(axis=1)))
    t = s["t0"] + np.arange(s["tj"].shape[1]) / 3600
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(t, s["t_amb"][d], color="0.5", lw=0.8, label="air")
    ax.plot(t, s["tj"][d], color="C0", lw=0.6, label="T_J base")
    ax.plot(t, h["tj"][d], color="C3", lw=0.6, label="T_J hard use")
    ax.set_xlabel("local time (h)"); ax.set_ylabel("°C"); ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "f2_day_delhi.png"), dpi=200); plt.close(fig)

    # 3 damage ratio to Madrid: (a) riding cycles only, (b) whole day; bars = base life model, lines = spread over 36 models
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    x = np.arange(len(CITIES))
    for ax, key, title in [(axs[0], "active_ratio_to_Madrid", "(a) riding cycles only"),
                           (axs[1], "ratio_to_Madrid", "(b) whole day incl. air-temperature swing")]:
        for j, p in enumerate(PROFILES):
            rows = out["B_life_model"][p]
            base = [r for r in rows if r["n"] == 5.0 and r["Ea"] == 0.617 and r["dT_min"] == 2.0][0]
            lo = [min(r[key][c] for r in rows) for c in CITIES]
            hi = [max(r[key][c] for r in rows) for c in CITIES]
            b = [base[key][c] for c in CITIES]
            ax.bar(x + (j - 0.5) * 0.38, b, 0.36, label=p, color=["C0", "C1"][j])
            ax.vlines(x + (j - 0.5) * 0.38, lo, hi, color="k", lw=0.8)
        ax.axhline(1, color="k", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(CITIES); ax.set_title(title, fontsize=8)
    axs[0].set_ylabel("damage relative to Madrid"); axs[0].legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "f3_ratio.png"), dpi=200); plt.close(fig)

    # 4 factors, delivery, Delhi
    rows = out["C_factors_delivery"]["Delhi"]
    base = rows[0]["D_index_base"]
    names = [r["factor"] for r in rows[1:]]
    vals = [r["D_index_base"] / base for r in rows[1:]]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.barh(np.arange(len(names)), np.array(vals)[order], color="C0")
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(np.array(names)[order], fontsize=6)
    ax.set_xscale("log"); ax.axvline(1, color="k", lw=0.5); ax.set_xlabel("damage relative to base (Delhi, delivery)")
    fig.savefig(os.path.join(FIG, "f4_factors.png"), dpi=200); plt.close(fig)

    # 5 rainflow histogram Delhi vs Madrid delivery
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    bins = np.arange(0, 41, 2)
    for c in ("Delhi", "Madrid", "Jakarta"):
        cy = cycles[("delivery", c)]
        ax.hist(cy[:, 0], bins=bins, weights=cy[:, 2], histtype="step", color=col[c], label=c)
    ax.set_yscale("log"); ax.set_xlabel("cycle range ΔT_J (K)"); ax.set_ylabel("cycles per year"); ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "f5_rainflow.png"), dpi=200); plt.close(fig)


def write_summary(out):
    L = []
    L.append("WEATHER 2025 (NASA POWER T2M)")
    for c, s in out["weather"].items():
        L.append(f"  {c:8s} mean {s['mean']:.2f}  p99 {s['p99']:.2f}  max {s['max']:.2f}  hours>35C {s['hours_above_35']}")
    for p in PROFILES:
        L.append(f"A BASE, {p}")
        for c in CITIES:
            m = out["A_base"][f"{p}/{c}"]
            L.append(f"  {c:8s} km/day {m['km_per_day']:.1f}  Tj max {m['tj_max']:.1f}  Tj p99 {m['tj_p99_riding']:.1f}  "
                     f"rise mean {m['rise_mean_riding']:.1f}  cycles>=2K {m['n_cycles_ge2K']}  D {m['D_index_base']:.3e} (n=3: {m['D_index_n3']:.2e})")
        rows = out["B_life_model"][p]
        L.append(f"B ratio to Madrid over {len(rows)} life models, {p}")
        for c in CITIES:
            r = [x["ratio_to_Madrid"][c] for x in rows]
            b = [x for x in rows if x["n"] == 5.0 and x["Ea"] == 0.617 and x["dT_min"] == 2.0][0]["ratio_to_Madrid"][c]
            L.append(f"  {c:8s} base {b:.2f}  min {min(r):.2f}  max {max(r):.2f}")
        first = [x["order"][0] for x in rows]
        L.append("  most damaged city by model: " + ", ".join(f"{c} {first.count(c)}" for c in CITIES if first.count(c)))
        L.append(f"B' riding cycles only (rainflow per ride), ratio to Madrid, {p}")
        for c in CITIES:
            r = [x["active_ratio_to_Madrid"][c] for x in rows]
            b = [x for x in rows if x["n"] == 5.0 and x["Ea"] == 0.617 and x["dT_min"] == 2.0][0]["active_ratio_to_Madrid"][c]
            L.append(f"  {c:8s} base {b:.2f}  min {min(r):.2f}  max {max(r):.2f}")
        first = [x["active_order"][0] for x in rows]
        L.append("  most damaged city by model: " + ", ".join(f"{c} {first.count(c)}" for c in CITIES if first.count(c)))
    L.append("C FACTORS, delivery (damage x base, Tj max, Ths max, riding h with Ths>80C)")
    for city, rows in out["C_factors_delivery"].items():
        b = rows[0]["D_index_base"]
        for r in rows:
            L.append(f"  {city:8s} {r['factor']:45s} x{r['D_index_base'] / b:8.2f}  Tj {r['tj_max']:.1f}  Ths {r['th_max']:.1f}  "
                     f"h>80 {r['riding_hours_th_above_80']:.1f}")
    L.append(f"D HARD USE {out['D_hard']['settings']}")
    for p in PROFILES:
        for c in CITIES:
            m = out["D_hard"]["results"][f"{p}/{c}"]
            L.append(f"  {p:9s} {c:8s} Tj max {m['tj_max']:.1f}  Ths max {m['th_max']:.1f}  h Ths>80 {m['riding_hours_th_above_80']:.1f}  "
                     f"D {m['D_index_base']:.3e} (n=3: {m['D_index_n3']:.2e}, n=6: {m['D_index_n6']:.2e})  ratio {out['D_hard_ratio_to_Madrid'][p][c]:.2f}")
    with open(os.path.join(RES, "summary.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
