"""Consistency checks of the input data and models. Run: python tests/test_models.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import wmtc, vehicle, mosfet, rainflow, weather  # noqa: E402


def test_wmtc_part1():
    v = wmtc.load("part1")
    assert len(v) == 600 and v.max() == 60.0
    assert abs(v.sum() / 3600 - 4.065) < 0.001          # km


def test_road_load_matches_gtr_table():
    a, b, _ = vehicle.road_load(0.0)
    assert (a, b) == (15.8, 0.0227)                      # GTR No. 2 Annex 3, 175 < m_ref <= 185 kg


def test_energy_plausible():
    # spec: 2.52 kWh for a claimed 120 km, about 21 Wh/km at the pack
    v, tq = vehicle.follow(wmtc.load("part1"))
    p = vehicle.mech_power(v, tq)
    wh_km = np.clip(p, 0, None).sum() / 3600 / (v.sum() / 3600)
    assert 12 < wh_km < 25


def test_rds_normalised_at_25C():
    assert abs(mosfet.rds(25.0) - mosfet.RDS_TYP) < 1e-12


def test_reversals_fast_equals_simple():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=3000))
    for h in (0.0, 0.5, 2.0):
        out = [x[0]]; d0 = 0
        for v in x[1:]:
            d = v - out[-1]
            if d0 == 0:
                if abs(d) > h:
                    d0 = 1 if d > 0 else -1; out.append(v)
            elif d0 * d > 0:
                out[-1] = v
            elif abs(d) > h:
                d0 = -d0; out.append(v)
        assert np.allclose(rainflow.reversals(x, h), out)


def test_weather_complete():
    for c in weather.CITIES:
        t, T = weather.city_temperature(c)
        assert len(T) == 8760 and np.all(np.isfinite(T))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("all model tests passed")
