"""
WMTC (World-wide Motorcycle Test Cycle) speed traces, read from the official
text of UN GTR No. 2 (ECE/TRANS/180/Add.2, Annex 5, Tables A5-1 onwards).

The PDF itself is not redistributed here. To rebuild data/wmtc.csv:
    python wmtc.py path/to/GTR-No-2.pdf
This runs `pdftotext -layout` and parses every "time  normal  reduced" triple
under the headings "Table A5-x: Cycle part N". The parser checks that each
part has exactly 600 consecutive seconds.

Which part applies: GTR No. 2 (para 6.3, 6.5.4.1) classifies by engine
capacity and maximum speed. An electric scooter has no engine capacity; by
maximum speed alone, a 70 km/h scooter falls in the 50-100 km/h band of
subclass 1-3, which runs part 1 (normal speed). That mapping is our choice
and is stated in the report.
"""
import os
import re
import subprocess
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "wmtc.csv")


def parse_pdf(pdf_path):
    txt = subprocess.run(["pdftotext", "-layout", pdf_path, "-"], capture_output=True, text=True, check=True).stdout
    lines = txt.split("\n")
    start = next(i for i, L in enumerate(lines) if "DRIVING CYCLES FOR TYPE I TESTS" in L)
    part, data = None, {}
    for L in lines[start:]:
        m = re.search(r"Table A5-\d+:\s*Cycle part (\d)", L)
        if m:
            part = int(m.group(1))
            data.setdefault(part, {})
            continue
        if part is None:
            continue
        for t, a, b in re.findall(r"(?:^|\s)(\d{1,3})\s+(\d{1,3}\.\d)\s+(\d{1,3}\.\d)", L):
            data[part][int(t)] = (float(a), float(b))
    for p, d in data.items():
        if sorted(d) != list(range(1, 601)):
            raise ValueError(f"part {p}: expected seconds 1..600")
    return data


def write_csv(data, path=CSV):
    with open(path, "w") as f:
        f.write("# WMTC from UN GTR No. 2 (ECE/TRANS/180/Add.2), Annex 5. Speeds in km/h, 1 s steps.\n")
        f.write("t_s,part1,part1_reduced,part2,part2_reduced,part3,part3_reduced\n")
        for t in range(1, 601):
            row = [t] + [data[p][t][k] for p in (1, 2, 3) for k in (0, 1)]
            f.write(",".join(str(x) for x in row) + "\n")


def load(column="part1"):
    a = np.genfromtxt(CSV, delimiter=",", names=True, skip_header=1)
    return a[column].astype(float)


if __name__ == "__main__":
    d = parse_pdf(sys.argv[1])
    write_csv(d)
    for p in (1, 2, 3):
        v = np.array([d[p][t][0] for t in range(1, 601)])
        print(f"part {p}: max {v.max():.1f} km/h, distance {v.sum() / 3600:.3f} km")
