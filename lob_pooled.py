"""
Pooled version of the LOB imbalance experiment (several tickers together).

Put this file in the SAME folder as lob_imbalance_experiment.py (it imports from it).

Run (Windows, use `python`):
    python lob_pooled.py --tickers aapl_data amzn_data goog_data intc_data msft_data --out pooled_out

How pooling works (all steps are per ticker first, then combined):
  1. Each ticker is analysed on its own: chronological split (morning = train,
     afternoon = test), 1-second grid, same features and horizons as before.
  2. Future returns are divided by that ticker's TRAINING return std, so
     slopes from different stocks are comparable (units: std of the return per
     unit of imbalance).
  3. Volatility regimes are defined per ticker. Default --regime-cut day uses the
     whole day's trailing-volatility percentiles (33rd, 67th, 90th), so the
     extreme group (top 10%) has enough seconds in the test half.
     Only past volatility is used, never the target. --regime-cut train uses the
     training half only (stricter, but the extreme group gets very small).
  4. Combining across tickers:
       slopes        inverse-variance weighted average (Newey-West variances)
       hit rate      pooled counts; effective sample size = sum of n/h
       out-of-sample R2  1 - total squared error / total variance
       extreme-vs-low slope difference   inverse-variance combined
  5. CAUTION: the five stocks trade on the SAME day and move together, so they
     are not five independent samples. Pooled intervals are optimistic. Say this
     in the write-up.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import lob_imbalance_experiment as base


def run_ticker(name, folder, regime_cut):
    frames = base.load_folder(folder)
    data = base.build_dataset(frames)
    rows = []
    for feature in ["I1", "I5"]:
        for h in base.HORIZONS:
            ycol = f"y{h}"
            d = data[["order", feature, ycol, "pastvol"]].dropna()
            if len(d) < 1000:
                continue
            split = d["order"].quantile(0.5)
            train = d[d["order"] < split - base.GAP]
            test = d[d["order"] >= split]
            if len(train) < 500 or len(test) < 500:
                continue

            ref = d["pastvol"] if regime_cut == "day" else train["pastvol"]
            cuts = np.percentile(ref, [33.3, 66.7, 90.0])

            sd = train[ycol].std()
            if not sd > 0:
                continue
            xtr = train[feature].to_numpy()
            ytr = train[ycol].to_numpy() / sd
            slope_tr, icpt_tr = np.polyfit(xtr, ytr, 1)
            ybar = ytr.mean()

            xte = test[feature].to_numpy()
            yte = test[ycol].to_numpy() / sd
            reg = base.assign_regime(test["pastvol"].to_numpy(), cuts)

            cols = [(reg == k).astype(float) for k in range(4)]
            cols += [xte * (reg == k) for k in range(4)]
            X = np.column_stack(cols)
            present = [k for k in range(4) if (reg == k).sum() > 50]
            keep = present + [4 + k for k in present]
            beta, V = base.ols_hac(X[:, keep], yte, lag=h)
            pos = {k: i for i, k in enumerate(present)}
            np_ = len(present)

            c_val, c_var = np.nan, np.nan
            if 0 in pos and 3 in pos:
                c = np.zeros(len(keep))
                c[np_ + pos[3]] = 1.0
                c[np_ + pos[0]] = -1.0
                c_val, c_var = float(c @ beta), float(c @ V @ c)

            for k in present:
                m = reg == k
                i = np_ + pos[k]
                pred = icpt_tr + slope_tr * xte[m]
                sse = float(np.sum((yte[m] - pred) ** 2))
                sst = float(np.sum((yte[m] - ybar) ** 2))
                nz = m & (yte != 0) & (xte != 0)
                hits = int(np.sum(np.sign(xte[nz]) == np.sign(yte[nz])))
                rows.append({
                    "ticker": name, "feature": feature, "horizon_s": h,
                    "regime": base.REGIMES[k], "n_test": int(m.sum()),
                    "slope": float(beta[i]), "slope_var": float(V[i, i]),
                    "sse": sse, "sst": sst, "hits": hits, "n_nz": int(nz.sum()),
                    "c_val": c_val if k == 3 else np.nan,
                    "c_var": c_var if k == 3 else np.nan,
                })
    return rows


def pool(per):
    out = []
    for (feat, h, reg), g in per.groupby(["feature", "horizon_s", "regime"]):
        w = 1.0 / g["slope_var"].replace(0, np.nan)
        w = w.fillna(0)
        if w.sum() > 0:
            slope = float((w * g["slope"]).sum() / w.sum())
            se = float(1.0 / np.sqrt(w.sum()))
            t = slope / se
        else:
            slope = se = t = np.nan

        n_nz = int(g["n_nz"].sum())
        hits = int(g["hits"].sum())
        n_eff = n_nz / h
        hit = hits / n_nz if n_nz else np.nan
        half = 1.96 * 0.5 / np.sqrt(max(n_eff, 1.0)) if n_nz else np.nan
        sst = g["sst"].sum()
        r2 = 100 * (1 - g["sse"].sum() / sst) if sst > 0 else np.nan

        ct = np.nan
        gc = g.dropna(subset=["c_val", "c_var"])
        gc = gc[gc["c_var"] > 0]
        if reg == "extreme" and len(gc):
            wc = 1.0 / gc["c_var"]
            cv = float((wc * gc["c_val"]).sum() / wc.sum())
            ct = cv / float(1.0 / np.sqrt(wc.sum()))

        out.append({
            "feature": feat, "horizon_s": h, "regime": reg,
            "tickers": int(g["ticker"].nunique()), "n_test": int(g["n_test"].sum()),
            "n_effective": round(n_eff, 1),
            "slope_std_per_unit": slope, "slope_t": t,
            "oos_R2_pct": r2, "hit_rate": hit, "hit_ci_halfwidth": half,
            "extreme_minus_low_t": ct,
        })
    res = pd.DataFrame(out)
    order = {r: i for i, r in enumerate(base.REGIMES)}
    res["_o"] = res["regime"].map(order)
    res = res.sort_values(["feature", "horizon_s", "_o"]).drop(columns="_o")
    return res.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True,
                    help="data folders, one per ticker (each holds its message + orderbook csv)")
    ap.add_argument("--out", default="pooled_out")
    ap.add_argument("--regime-cut", choices=["day", "train"], default="day")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_rows = []
    for folder in args.tickers:
        name = os.path.basename(os.path.normpath(folder)).replace("_data", "").upper()
        print(f"\n=== {name} ===")
        all_rows += run_ticker(name, folder, args.regime_cut)
    if not all_rows:
        sys.exit("Not enough data to analyse.")

    per = pd.DataFrame(all_rows)
    per["hit_rate"] = per["hits"] / per["n_nz"].replace(0, np.nan)
    per.to_csv(os.path.join(args.out, "per_ticker_results.csv"), index=False)

    res = pool(per)
    res.to_csv(os.path.join(args.out, "pooled_results.csv"), index=False)
    base.make_plot(res, os.path.join(args.out, "pooled_imbalance_by_horizon_and_vol.png"))

    pd.set_option("display.width", 220, "display.max_columns", 30)
    show = res.copy()
    for c in ["slope_std_per_unit", "slope_t", "oos_R2_pct", "extreme_minus_low_t",
              "hit_rate", "hit_ci_halfwidth"]:
        show[c] = show[c].round(3)
    print(f"\nPOOLED RESULTS (test half, regime cut = {args.regime_cut})")
    print(show.to_string(index=False))

    print("\nPER-TICKER hit rate, level-1 imbalance (I1), horizon 1 s and 10 s")
    for h in (1, 10):
        sub = per[(per["feature"] == "I1") & (per["horizon_s"] == h)]
        piv = sub.pivot(index="ticker", columns="regime", values="hit_rate")
        piv = piv[[r for r in base.REGIMES if r in piv.columns]].round(3)
        print(f"\nh = {h} s")
        print(piv.to_string())

    print(f"\nSaved to {args.out}: pooled_results.csv, per_ticker_results.csv, PNG")
    print("Caution: the stocks share the same day and move together, so pooled "
          "intervals are optimistic.")


if __name__ == "__main__":
    main()
