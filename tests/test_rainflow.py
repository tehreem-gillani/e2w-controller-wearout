"""Checks for the rainflow counter. Run: python -m pytest tests (or python tests/test_rainflow.py)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import rainflow  # noqa: E402


def test_astm_example():
    # Small load history -2, 1, -3, 5, -1, 3, -4, 4, -2.
    # Hand count with the three-point method:
    #   half cycles: 3 (-2..1), 4 (1..-3), 8 (-3..5), 9 (5..-4), 8 (-4..4), 6 (4..-2)
    #   full cycle : 4 (-1..3)
    c = rainflow.count([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    full = sorted(r for r, m, n in c if n == 1.0)
    half = sorted(r for r, m, n in c if n == 0.5)
    assert full == [4]
    assert half == [3, 4, 6, 8, 8, 9]


def test_constant_amplitude():
    # 10 identical cycles between 0 and 10 -> about 10 cycles of range 10
    x = np.tile([0.0, 10.0], 10).tolist() + [0.0]
    c = rainflow.count(x)
    n10 = sum(n for r, m, n in c if abs(r - 10) < 1e-9)
    assert abs(n10 - 10.0) < 1e-9


def test_hysteresis_drops_ripple():
    x = [0, 10, 9.8, 10, 0]
    assert len(rainflow.reversals(x, hysteresis=0.5)) == 3


if __name__ == "__main__":
    test_astm_example(); test_constant_amplitude(); test_hysteresis_drops_ripple()
    print("all rainflow tests passed")
