#!/usr/bin/env python3
"""Paired A/B statistics for tools/energy_ab.sh and tools/energy_warm_ab.py rows.

Per metric: mean of B - A over complete pairs, its 95% interval, and the smallest
effect (as a share of A's mean) the run could have detected with 80% power. A second
table fits y = a + b*isB + c*t over all runs in file order, which absorbs a linear
drift (battery, temperature). When rows carry a `variant` key (energy_warm_ab.py
with --swap-after), "A" is the variant that sat in slot A at the start, "B" is the
other one, and the fit adds a daemon-slot term, so a slot bias such as the 1.9% wall
RTF gap seen between two daemons running the same binary is not read as an effect.
The position check compares first and second runs of a pair.
"""

import math
import statistics
import sys

# Two-sided 95% t quantiles for 1..30 degrees of freedom.
T95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]
Z_POWER80 = 0.8416
Z_ALPHA = 1.96

METRICS = [
    ("cpu_s", "process CPU s"),
    ("j_per_audio_s", "J / audio s"),
    ("mean_w", "mean W"),
    ("wall_rtf", "wall RTF"),
    ("peak_c", "peak gold C"),
    ("idle_w", "idle W"),
]


def tcrit(df):
    return T95[df - 1] if df <= 30 else Z_ALPHA


def ols(X, y):
    """Least squares by Gauss-Jordan on the normal equations: (beta, se, df, resid sd)."""
    n, k = len(X), len(X[0])
    if n <= k:
        return None
    m = [[sum(X[r][i] * X[r][j] for r in range(n)) for j in range(k)]
         + [1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-9:
            return None
        m[c], m[p] = m[p], m[c]
        piv = m[c][c]
        m[c] = [v / piv for v in m[c]]
        for r in range(k):
            if r != c:
                f = m[r][c]
                m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    inv = [row[k:] for row in m]
    xty = [sum(X[r][i] * y[r] for r in range(n)) for i in range(k)]
    beta = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]
    sse = sum((y[r] - sum(X[r][i] * beta[i] for i in range(k))) ** 2 for r in range(n))
    s2 = sse / (n - k)
    return beta, [math.sqrt(s2 * inv[i][i]) for i in range(k)], n - k, math.sqrt(s2)


def parse(path):
    rows = []
    for line in open(path):
        fields = dict(w.split("=", 1) for w in line.split() if "=" in w)
        if fields.get("rc") == "0" and "pair" in fields:
            rows.append(fields)
    # With variants, arm becomes "which variant" and the original slot is kept.
    if rows and all("variant" in r for r in rows):
        baseline = next((r["variant"] for r in rows if r["arm"] == "A"), rows[0]["variant"])
        for r in rows:
            r["slot"] = r["arm"]
            r["arm"] = "A" if r["variant"] == baseline else "B"
    return rows


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit("usage: energy_ab_analyze.py rows.log [--drop-suspect]")
    rows = parse(args[0])
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], {})[r["arm"]] = r
    pairs = [p for p in by_pair.values() if "A" in p and "B" in p]
    n = len(pairs)
    print(f"complete pairs: {n} (rows: {len(rows)})")
    if rows and "variant" in rows[0]:
        names = {r["arm"]: r["variant"] for r in rows}
        print(f"A = {names.get('A')}, B = {names.get('B')} (slot swap handled in the fit)")
    if n < 2:
        print("need at least 2 complete pairs")
        return
    short = [r["config"] for r in rows if number(r, "samples") is not None and number(r, "real_s") is not None
             and number(r, "samples") < 0.8 * number(r, "real_s") * (number(r, "hz") or 1)]
    if short:
        print("possible suspend (fewer samples than seconds):", ", ".join(short))
    idles = sorted(v for v in (number(r, "idle_w") for r in rows) if v is not None)
    idle_med = idles[len(idles) // 2] if idles else 0
    suspect = [f"{r['config']} (stall {number(r, 'stall_s'):.0f}s)" for r in rows if (number(r, "stall_s") or 0) > 5]
    suspect += [f"{r['config']} (idle {number(r, 'idle_w'):.2f} W, median {idle_med:.2f})" for r in rows
                if (number(r, "idle_w") or 0) > max(1.0, 3 * idle_med)]
    if suspect:
        print("SUSPECT runs (process stalled, or the phone was in use beforehand):", "; ".join(suspect))
    if "--drop-suspect" in sys.argv:
        bad = {r["pair"] for r in rows if (number(r, "stall_s") or 0) > 5
               or (number(r, "idle_w") or 0) > max(1.0, 3 * idle_med)}
        rows = [r for r in rows if r["pair"] not in bad]
        by_pair = {}
        for r in rows:
            by_pair.setdefault(r["pair"], {})[r["arm"]] = r
        pairs = [p for p in by_pair.values() if "A" in p and "B" in p]
        n = len(pairs)
        print(f"--drop-suspect: dropped pairs {sorted(bad, key=int)}; {n} pairs remain")
    print(f"{'metric':<18}{'mean A':>10}{'B-A %':>9}{'95% CI %':>20}{'sd(B-A) %':>11}{'MDE80 %':>9}{'pairs for 3%':>14}{'for 5%':>8}")
    for key, label in METRICS:
        both = [(number(p["A"], key), number(p["B"], key)) for p in pairs]
        both = [(a, b) for a, b in both if a is not None and b is not None]
        if len(both) < 2:
            continue
        a_mean = statistics.mean(a for a, _ in both)
        if a_mean == 0:
            continue
        d = [b - a for a, b in both]
        m = len(d)
        d_mean = statistics.mean(d)
        d_sd = statistics.stdev(d)
        se = d_sd / math.sqrt(m)
        half = tcrit(m - 1) * se
        mde = (tcrit(m - 1) + Z_POWER80) * se
        need = lambda eff: math.ceil(((Z_ALPHA + Z_POWER80) * d_sd / (eff * a_mean)) ** 2)
        pct = lambda x: 100 * x / a_mean
        print(f"{label:<18}{a_mean:>10.3f}{pct(d_mean):>+9.2f}"
              f"{'[%+.2f, %+.2f]' % (pct(d_mean - half), pct(d_mean + half)):>20}"
              f"{pct(d_sd):>11.2f}{pct(mde):>9.2f}{need(0.03):>14}{need(0.05):>8}")
    print()
    use_slot = bool(rows) and all("slot" in r for r in rows) and len({(r["slot"], r["arm"]) for r in rows}) == 4
    cols = "y = a + b*isB + c*t" + (" + s*slotB" if use_slot else "")
    print(f"arm effect with a linear time drift, all runs in file order ({cols}):")
    print(f"{'metric':<18}{'B effect %':>11}{'95% CI %':>20}{'drift %/run':>12}"
          + (f"{'slot B %':>10}" if use_slot else "")
          + f"{'resid sd %':>11}{'MDE80 %':>9}{'pairs 3%':>9}{'pairs 5%':>9}")
    for key, label in METRICS:
        used = [(i, r) for i, r in enumerate(rows) if number(r, key) is not None and r["pair"] in by_pair
                and "A" in by_pair[r["pair"]] and "B" in by_pair[r["pair"]]]
        if len(used) < 6:
            continue
        mid = (len(used) - 1) / 2
        X = [[1.0, 1.0 if r["arm"] == "B" else 0.0, k - mid]
             + ([1.0 if r["slot"] == "B" else 0.0] if use_slot else []) for k, (_, r) in enumerate(used)]
        y = [number(r, key) for _, r in used]
        fit = ols(X, y)
        if fit is None:
            continue
        beta, se, df, resid_sd = fit
        a_mean = statistics.mean(v for v, x in zip(y, X) if x[1] == 0.0)
        if a_mean == 0:
            continue
        half = tcrit(df) * se[1]
        mde = (tcrit(df) + Z_POWER80) * se[1]
        npairs = len(used) / 2
        need = lambda eff: math.ceil(npairs * ((Z_ALPHA + Z_POWER80) * se[1] / (eff * a_mean)) ** 2)
        print(f"{label:<18}{100 * beta[1] / a_mean:>+11.2f}"
              f"{'[%+.2f, %+.2f]' % (100 * (beta[1] - half) / a_mean, 100 * (beta[1] + half) / a_mean):>20}"
              f"{100 * beta[2] / a_mean:>+12.3f}"
              + (f"{100 * beta[3] / a_mean:>+10.2f}" if use_slot else "")
              + f"{100 * resid_sd / a_mean:>11.2f}{100 * mde / a_mean:>9.2f}{need(0.03):>9}{need(0.05):>9}")
    print()
    print("position effect (second run minus first run, share of the first run's mean):")
    for key, label in METRICS[:3]:
        diffs = []
        firsts = []
        for p in by_pair.values():
            runs = sorted(p.values(), key=lambda r: int(r["pos"]))
            if len(runs) == 2:
                a, b = number(runs[0], key), number(runs[1], key)
                if a is not None and b is not None:
                    diffs.append(b - a)
                    firsts.append(a)
        if len(diffs) >= 2 and statistics.mean(firsts):
            se = statistics.stdev(diffs) / math.sqrt(len(diffs))
            t = f"{statistics.mean(diffs) / se:+.2f}" if se else "n/a"
            print(f"  {label:<18}{100 * statistics.mean(diffs) / statistics.mean(firsts):>+8.2f} %  (t = {t})")


if __name__ == "__main__":
    main()
