"""
Hourly air temperature (T2M, MERRA-2) from NASA POWER for 2025, five cities.

Time: the files say time-standard=LST. In Project A (Karachi) a clear-sky
fit showed POWER's LST is the nominal zone of the longitude, round(lon/15)
hours from UTC, not the civil time zone. We apply the same rule to every
city and shift to the local clock:
    clock = LST + (civil UTC offset - round(lon/15))
Madrid uses CET/CEST (summer time 30 Mar - 26 Oct 2025). This only moves
the riding hours by up to 1 h against the temperature, and is stated as an
assumption in the report.
"""
import datetime as dt
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NASA = os.path.join(HERE, "..", "data", "nasa")

CITIES = {
    #  name       file                lon        civil UTC offset (h)
    "Karachi": ("karachi_2025.csv", 67.0011, 5.0),
    "Delhi":   ("delhi_2025.csv", 77.2090, 5.5),
    "Jakarta": ("jakarta_2025.csv", 106.8456, 7.0),
    "Lagos":   ("lagos_2025.csv", 3.3792, 1.0),
    "Madrid":  ("madrid_2025.csv", -3.7038, None),   # CET/CEST
}
# summer time 30 Mar 01:00 UTC to 26 Oct 01:00 UTC, written here in CET (UTC+1)
DST_2025 = (dt.datetime(2025, 3, 30, 2), dt.datetime(2025, 10, 26, 2))


def _read(path):
    rows, cols = [], None
    with open(path) as f:
        data = False
        for line in f:
            if line.startswith("-END HEADER-"):
                data = True
                continue
            if not data:
                continue
            if line.startswith("YEAR"):
                cols = line.strip().split(",")
                continue
            rows.append([float(x) for x in line.strip().split(",")])
    a = np.array(rows)
    c = {name: i for i, name in enumerate(cols)}         # column order differs between files
    return {"year": a[:, c["YEAR"]].astype(int), "month": a[:, c["MO"]].astype(int),
            "day": a[:, c["DY"]].astype(int), "hour": a[:, c["HR"]].astype(int), "t2m": a[:, c["T2M"]]}


def city_temperature(name):
    """Returns (t_hours_local, T2M). t_hours_local[i] = hours since
    1 Jan 2025 00:00 local clock at the middle of hourly value i."""
    fname, lon, utc = CITIES[name]
    w = _read(os.path.join(NASA, fname))
    lst_zone = round(lon / 15.0)
    base = np.array([(dt.datetime(y, m, d, h) - dt.datetime(2025, 1, 1)).total_seconds() / 3600
                     for y, m, d, h in zip(w["year"], w["month"], w["day"], w["hour"])])
    if utc is None:
        local = []
        for b in base:
            t_utc = dt.datetime(2025, 1, 1) + dt.timedelta(hours=float(b) - lst_zone)
            off = 2.0 if DST_2025[0] <= t_utc + dt.timedelta(hours=1) < DST_2025[1] else 1.0
            local.append(b - lst_zone + off)
        shift = np.array(local)
    else:
        shift = base + (utc - lst_zone)
    t2m = w["t2m"]
    if np.any(t2m < -900):
        raise ValueError(f"{name}: missing values in T2M")
    return shift + 0.5, t2m


def ambient_at(t_local_h, t_hours, t2m):
    """Linear interpolation of the hourly temperature."""
    return np.interp(t_local_h, t_hours, t2m)


def summary():
    out = {}
    for c in CITIES:
        t, T = city_temperature(c)
        out[c] = {"mean": float(T.mean()), "p99": float(np.percentile(T, 99)), "max": float(T.max()),
                  "hours_above_35": int((T > 35).sum())}
    return out


if __name__ == "__main__":
    for c, s in summary().items():
        print(c, {k: round(v, 1) for k, v in s.items()})
