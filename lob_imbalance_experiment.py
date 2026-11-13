"""
LOB imbalance experiment: does order book imbalance still predict mid-price
changes at longer horizons, and in high-volatility periods?

Written for the research question in Prof. Kyungsub Lee's reply (12 Nov 2026).

What it does
------------
1. Loads LOBSTER-format files (message file + orderbook file) for one or more days.
2. Samples the book on a 1-second grid (last known state at or before each second).
3. Builds two imbalance features known at time t:
       I1 = (bid vol - ask vol) / (bid vol + ask vol) at level 1
       I5 = same, summed over the first 5 levels
4. Target = FUTURE mid-price change over h seconds (bps). Only future prices go
   into the target, so no past information leaks into it.
5. Volatility regime at time t = std of the past 60 one-second mid returns
   (past data only). Thresholds come from the TRAIN half only.
       low (<p33), mid (p33-p67), high (p67-p90), extreme (>=p90)
6. Chronological split: first 50% of the grid = train, last 50% = test.
7. For each feature and horizon, on the TEST half only:
       - slope of future return on imbalance in each regime (Newey-West
         standard errors, lag = h, because h-second returns overlap)
       - out-of-sample R2 of the train-fitted pooled slope, per regime
       - directional hit rate (sign of imbalance vs sign of return, nonzero
         cases only), with an overlap-adjusted 95% interval
       - t-test of (slope in extreme regime - slope in low regime)

Run (Windows, use `python`):
    python lob_imbalance_experiment.py --make-synthetic synth_data
    python lob_imbalance_experiment.py --data synth_data --out synth_out
    python lob_imbalance_experiment.py --data real_data --out real_out

--data is a folder holding pairs of LOBSTER files, for example
    AAPL_2012-06-21_34200000_57600000_message_10.csv
    AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
Several days are concatenated in date order.

Requires only numpy, pandas, matplotlib.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HORIZONS = [1, 5, 10, 30, 60, 300]  # seconds
VOL_WINDOW = 60                      # seconds of past returns for the vol regime
REGIMES = ["low", "mid", "high", "extreme"]
GAP = 300                            # seconds dropped at the train/test boundary


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_lobster_pair(msg_path, book_path):
    """Return (times_in_seconds, book DataFrame) for one LOBSTER day."""
    msg = pd.read_csv(msg_path, header=None, usecols=[0], names=["t"])
    book = pd.read_csv(book_path, header=None)
    if len(msg) != len(book):
        raise ValueError(
            f"Row mismatch: {os.path.basename(msg_path)} has {len(msg)} rows, "
            f"{os.path.basename(book_path)} has {len(book)} rows"
        )
    if book.shape[1] % 4 != 0:
        raise ValueError("Orderbook file should have 4 columns per level")
    levels = book.shape[1] // 4
    cols = []
    for i in range(1, levels + 1):
        cols += [f"ap{i}", f"av{i}", f"bp{i}", f"bv{i}"]
    book.columns = cols
    return msg["t"].to_numpy(dtype=float), book


def day_grid_frame(times, book, levels_for_depth=5):
    """Sample one day on a 1-second grid and return a DataFrame of features."""
    t0 = int(np.ceil(times[0]))
    t1 = int(np.floor(times[-1]))
    grid = np.arange(t0, t1 + 1, dtype=float)
    idx = np.searchsorted(times, grid, side="right") - 1
    ok = idx >= 0
    grid, idx = grid[ok], idx[ok]
    b = book.iloc[idx].reset_index(drop=True)

    # LOBSTER marks empty levels with +/-9999999999; prices are dollars * 10000
    valid = (b["ap1"] < 1e9) & (b["bp1"] > -1e9) & (b["ap1"] > b["bp1"])
    mid = (b["ap1"] + b["bp1"]) / 2.0 / 10000.0

    bv1, av1 = b["bv1"].astype(float), b["av1"].astype(float)
    i1 = (bv1 - av1) / (bv1 + av1)

    depth = min(levels_for_depth, sum(c.startswith("bv") for c in b.columns))
    bvs = sum(b[f"bv{i}"].astype(float) for i in range(1, depth + 1))
    avs = sum(b[f"av{i}"].astype(float) for i in range(1, depth + 1))
    i5 = (bvs - avs) / (bvs + avs)

    df = pd.DataFrame({"t": grid, "mid": mid, "I1": i1, "I5": i5})
    df.loc[~valid.to_numpy(), ["mid", "I1", "I5"]] = np.nan
    return df


def load_folder(folder):
    msgs = sorted(glob.glob(os.path.join(folder, "*_message_*.csv")))
    if not msgs:
        sys.exit(f"No *_message_*.csv files found in {folder}")
    frames = []
    for k, mp in enumerate(msgs):
        bp = mp.replace("_message_", "_orderbook_")
        if not os.path.exists(bp):
            sys.exit(f"Missing orderbook file for {mp}")
        times, book = load_lobster_pair(mp, bp)
        df = day_grid_frame(times, book)
        df["day"] = k
        frames.append(df)
        print(f"  loaded {os.path.basename(mp)}: {len(df)} one-second rows")
    return frames


# ----------------------------------------------------------------------------
# Features, targets, regimes
# ----------------------------------------------------------------------------
def add_targets_and_vol(df):
    """Per-day: future returns (bps) and past realised vol. Never crosses days."""
    out = df.copy()
    logmid = np.log(out["mid"])
    r1 = logmid.diff() * 1e4
    out["pastvol"] = r1.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    for h in HORIZONS:
        out[f"y{h}"] = (logmid.shift(-h) - logmid) * 1e4
    return out


def build_dataset(frames):
    parts = [add_targets_and_vol(f) for f in frames]
    data = pd.concat(parts, ignore_index=True)
    data["order"] = np.arange(len(data))
    return data


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def ols_hac(X, y, lag):
    """OLS with Newey-West (Bartlett) covariance. Returns beta, V."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * (n / max(n - k, 1))
    return beta, V


def assign_regime(vol, cuts):
    p33, p67, p90 = cuts
    r = np.full(len(vol), -1)
    r[vol < p33] = 0
    r[(vol >= p33) & (vol < p67)] = 1
    r[(vol >= p67) & (vol < p90)] = 2
    r[vol >= p90] = 3
    return r


def analyse(data, feature, h):
    """Return list of result rows (one per regime) for one feature and horizon."""
    ycol = f"y{h}"
    d = data[["order", "day", feature, ycol, "pastvol"]].dropna().copy()
    if len(d) < 1000:
        return []

    split_order = d["order"].quantile(0.5)
    train = d[d["order"] < split_order - GAP]
    test = d[d["order"] >= split_order]
    if len(train) < 500 or len(test) < 500:
        return []

    cuts = np.percentile(train["pastvol"], [33.3, 66.7, 90.0])
    train = train.assign(reg=assign_regime(train["pastvol"].to_numpy(), cuts))
    test = test.assign(reg=assign_regime(test["pastvol"].to_numpy(), cuts))

    # pooled slope fitted on TRAIN only (a practitioner's single model)
    xtr, ytr = train[feature].to_numpy(), train[ycol].to_numpy()
    Xtr = np.column_stack([np.ones(len(xtr)), xtr])
    b_tr = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    ybar_train = ytr.mean()

    # regime-specific slopes on TEST with Newey-West errors (one regression)
    xte, yte, rte = (test[feature].to_numpy(), test[ycol].to_numpy(),
                     test["reg"].to_numpy())
    cols = []
    for k in range(4):
        cols.append((rte == k).astype(float))
    for k in range(4):
        cols.append(xte * (rte == k))
    X = np.column_stack(cols)
    # drop regimes that are empty in the test half
    present = [k for k in range(4) if (rte == k).sum() > 50]
    keep = present + [4 + k for k in present]
    beta, V = ols_hac(X[:, keep], yte, lag=h)
    pos = {k: i for i, k in enumerate(present)}

    contrast_t = np.nan
    if 0 in pos and 3 in pos:
        c = np.zeros(len(keep))
        c[len(present) + pos[3]] = 1.0
        c[len(present) + pos[0]] = -1.0
        var = c @ V @ c
        if var > 0:
            contrast_t = (c @ beta) / np.sqrt(var)

    rows = []
    for k in range(4):
        m = rte == k
        n = int(m.sum())
        if k not in pos:
            continue
        slope = beta[len(present) + pos[k]]
        se = np.sqrt(V[len(present) + pos[k], len(present) + pos[k]])

        pred = b_tr[0] + b_tr[1] * xte[m]
        sse = np.sum((yte[m] - pred) ** 2)
        sst = np.sum((yte[m] - ybar_train) ** 2)
        oos_r2 = 1.0 - sse / sst if sst > 0 else np.nan

        nz = m & (yte != 0) & (xte != 0)
        n_nz = int(nz.sum())
        if n_nz > 0:
            hit = float(np.mean(np.sign(xte[nz]) == np.sign(yte[nz])))
            n_eff = max(n_nz / h, 1.0)
            half = 1.96 * 0.5 / np.sqrt(n_eff)
        else:
            hit, n_eff, half = np.nan, np.nan, np.nan

        rows.append({
            "feature": feature, "horizon_s": h, "regime": REGIMES[k],
            "n_test": n, "n_effective": round(n / h, 1),
            "slope_bps_per_unit": slope, "slope_t_NW": slope / se if se > 0 else np.nan,
            "oos_R2_pct": 100 * oos_r2,
            "hit_rate": hit, "hit_ci_halfwidth": half,
            "extreme_minus_low_t": contrast_t if k == 3 else np.nan,
        })
    return rows


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def make_plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = [f for f in ["I1", "I5"] if f in set(res["feature"])]
    fig, axes = plt.subplots(len(feats), 2, figsize=(11, 4 * len(feats)), squeeze=False)
    colors = {"low": "#2a9d8f", "mid": "#8d99ae", "high": "#e9a23b", "extreme": "#c0392b"}
    for r, f in enumerate(feats):
        sub = res[res["feature"] == f]
        for reg in REGIMES:
            s = sub[sub["regime"] == reg].sort_values("horizon_s")
            if s.empty:
                continue
            axes[r][0].plot(s["horizon_s"], s["oos_R2_pct"], "o-", color=colors[reg], label=reg)
            axes[r][1].errorbar(s["horizon_s"], s["hit_rate"], yerr=s["hit_ci_halfwidth"],
                                fmt="o-", color=colors[reg], label=reg, capsize=3)
        axes[r][0].axhline(0, color="k", lw=0.6)
        axes[r][1].axhline(0.5, color="k", lw=0.6)
        axes[r][0].set_title(f"{f}: out-of-sample R2 (%) by volatility regime")
        axes[r][1].set_title(f"{f}: directional hit rate (95% interval, overlap-adjusted)")
        for a in axes[r]:
            a.set_xscale("log")
            a.set_xlabel("horizon (seconds)")
        axes[r][0].legend(title="past vol")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Synthetic data (pipeline test ONLY, never report as a finding)
# ----------------------------------------------------------------------------
def make_synthetic(outdir, days=2, seed=7):
    """Write LOBSTER-format files from a toy simulator with a KNOWN structure:
    imbalance is AR(1) (fast decay) and pushes the price by a fixed amount, while
    noise volatility is clustered. So imbalance should matter at short horizons and
    matter less when volatility is high."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    step = 0.5
    n = int(23400 / step)
    for d in range(days):
        phi, beta_imp = 0.9, 0.0035
        z = np.zeros(n)
        g = np.zeros(n)
        eps = rng.standard_normal(n)
        eta = rng.standard_normal(n)
        zeta = rng.standard_normal(n)
        for i in range(1, n):
            z[i] = phi * z[i - 1] + 0.6 * eta[i]
            g[i] = 0.999 * g[i - 1] + 0.05 * zeta[i]
        imb = np.tanh(z)
        sigma = 0.0025 * np.exp(g)
        p = np.empty(n)
        p[0] = 100.0
        for i in range(1, n):
            p[i] = p[i - 1] + sigma[i] * eps[i] + beta_imp * imb[i - 1]
        ask = np.ceil(p * 100) / 100.0
        bid = ask - 0.01
        times = 34200 + step * np.arange(n)
        rows = np.zeros((n, 20), dtype=np.int64)  # 5 levels x (ap, av, bp, bv)
        base = 400.0
        for lvl in range(5):
            w = 1.0 / (lvl + 1)
            av = base * np.exp(-0.9 * w * imb + 0.25 * rng.standard_normal(n))
            bv = base * np.exp(+0.9 * w * imb + 0.25 * rng.standard_normal(n))
            rows[:, 4 * lvl + 0] = np.round((ask + 0.01 * lvl) * 10000).astype(np.int64)
            rows[:, 4 * lvl + 1] = np.maximum(av, 1).astype(np.int64)
            rows[:, 4 * lvl + 2] = np.round((bid - 0.01 * lvl) * 10000).astype(np.int64)
            rows[:, 4 * lvl + 3] = np.maximum(bv, 1).astype(np.int64)
        tag = f"SYN_2099-01-{d + 1:02d}_34200000_57600000"
        pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{tag}_orderbook_5.csv"),
                                  header=False, index=False)
        msg = pd.DataFrame({"t": times, "type": 1, "id": np.arange(n), "size": 100,
                            "price": rows[:, 0], "dir": 1})
        msg.to_csv(os.path.join(outdir, f"{tag}_message_5.csv"), header=False, index=False)
    print(f"Synthetic LOBSTER-format files written to {outdir} (toy data, pipeline test only)")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="folder with LOBSTER message/orderbook csv pairs")
    ap.add_argument("--out", default="lob_out", help="output folder")
    ap.add_argument("--make-synthetic", metavar="DIR", help="write toy data to DIR and exit")
    args = ap.parse_args()

    if args.make_synthetic:
        make_synthetic(args.make_synthetic)
        return
    if not args.data:
        ap.error("give --data FOLDER (or --make-synthetic DIR)")

    os.makedirs(args.out, exist_ok=True)
    print("Loading...")
    frames = load_folder(args.data)
    data = build_dataset(frames)
    print(f"Total one-second rows: {len(data)}")

    all_rows = []
    for feature in ["I1", "I5"]:
        for h in HORIZONS:
            all_rows += analyse(data, feature, h)
    if not all_rows:
        sys.exit("Not enough data to analyse. Use at least one full trading day.")
    res = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.out, "results.csv")
    res.to_csv(csv_path, index=False)
    make_plot(res, os.path.join(args.out, "imbalance_by_horizon_and_vol.png"))

    pd.set_option("display.width", 200, "display.max_columns", 20)
    show = res.copy()
    for c in ["slope_bps_per_unit", "slope_t_NW", "oos_R2_pct", "extreme_minus_low_t"]:
        show[c] = show[c].round(3)
    show["hit_rate"] = show["hit_rate"].round(3)
    show["hit_ci_halfwidth"] = show["hit_ci_halfwidth"].round(3)
    print("\nRESULTS (test half only)")
    print(show.to_string(index=False))
    print(f"\nSaved: {csv_path} and the PNG in {args.out}")
    print("Reminder: n_effective = n_test / horizon. Small values mean wide uncertainty.")


if __name__ == "__main__":
    main()
