# LOB imbalance, horizon and volatility

A small experiment I ran for a research proposal. The question: does limit-order-book
volume imbalance still predict price moves beyond very short horizons, and does it
hold up when volatility is extreme?

This is a first look, not a finished study. It uses one trading day.

## What I found

- Level-1 imbalance predicts the direction of the next-second mid-price move.
  In AAPL, AMZN and GOOG the hit rate is about 56% in calm periods and about 60%
  in the top 10% of volatility. I see no sign that high volatility weakens it.
- The edge fades fast. By about 30 seconds the hit rate is near 52% and the
  intervals include 50%.
- Five-level imbalance added nothing.
- INTC and MSFT give much higher hit rates (above 0.75 at 1 s). Their tick is
  large relative to their price, so I treat the three-stock numbers as the
  cautious case.

## Files

- `lob_imbalance_experiment.py`: the analysis for one stock (loads the data,
  builds imbalance and volatility, runs the tests, saves results and a plot).
- `lob_pooled.py`: runs several stocks and pools them. It imports the file above,
  so keep both in the same folder.
- `result_3stocks.csv`: pooled results for AAPL, AMZN, GOOG (the ones used in the proposal).
- `result_5stock.csv`: the same with INTC and MSFT added.

## Data

LOBSTER free sample files (Nasdaq, 21 June 2012; AAPL, AMZN, GOOG, INTC, MSFT;
10 levels), from the sample page on lobsterdata.com. The data is not included here.
Put each stock's message and orderbook CSV files in its own folder, such as `aapl_data`.

## How to run

One stock:

```
python lob_imbalance_experiment.py --data aapl_data --out aapl_out
```

Several stocks pooled:

```
python lob_pooled.py --t
