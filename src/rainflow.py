"""
Rainflow cycle counting, three-point method of ASTM E1049-85 (para 5.4.4).
Residual ranges are counted as half cycles. Written here instead of taken
from a package so every step can be read and tested (tests/test_rainflow.py).
"""
import numpy as np


def reversals(x, hysteresis=0.0):
    """Turning points of x. Changes smaller than `hysteresis` are ignored,
    the usual way to drop sensor-level ripple before counting."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.copy()
    # fast pre-pass: keep the end points and every local extremum
    d = np.diff(x)
    keep = d != 0
    xs = np.concatenate([[x[0]], x[1:][keep]])
    if len(xs) > 2:
        s = np.sign(np.diff(xs))
        turn = np.flatnonzero(s[1:] != s[:-1]) + 1
        xs = np.concatenate([[xs[0]], xs[turn], [xs[-1]]])
    x = xs
    out = [x[0]]
    direction = 0
    for v in x[1:]:
        d = v - out[-1]
        if direction == 0:
            if abs(d) > hysteresis:
                direction = 1 if d > 0 else -1
                out.append(v)
        elif direction * d > 0:
            out[-1] = v                      # still moving the same way: extend
        elif abs(d) > hysteresis:
            direction = -direction
            out.append(v)
    return np.array(out)


def count(x, hysteresis=0.0):
    """Returns a list of (range, mean, count) with count 1.0 or 0.5."""
    pts = list(reversals(x, hysteresis))
    stack = []
    cycles = []
    for p in pts:
        stack.append(p)
        while len(stack) >= 3:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])
            if X < Y:
                break
            if len(stack) == 3:
                cycles.append((Y, 0.5 * (stack[-2] + stack[-3]), 0.5))
                stack.pop(0)
            else:
                cycles.append((Y, 0.5 * (stack[-2] + stack[-3]), 1.0))
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for a, b in zip(stack[:-1], stack[1:]):
        cycles.append((abs(b - a), 0.5 * (a + b), 0.5))
    return cycles
