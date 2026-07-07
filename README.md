# FlightDelay

A model that tries to answer one question before a plane pushes back from the gate: is this flight going to land more than 15 minutes late (the FAA's own definition of "delayed")? Everything it uses is knowable at scheduling time — carrier, route, scheduled departure/arrival, distance, day/month — nothing about weather, gate conflicts, or air traffic control, which is exactly why it struggles with the hardest cases.

## Getting it running

```bash
pip install -r requirements.txt
python train.py --synthetic       # or: python train.py --csv path/to/flights.csv
streamlit run app.py
pytest tests/ -v
```

`train.py` writes `models/metrics.json`. The dashboard (`app.py`) trains its own models interactively in the browser, so it doesn't need that file to exist first — it re-fits whatever you ask for from the sidebar.

## Where the data comes from

Three options, picked in the sidebar:

- **nycflights13 (live)** — pulls the real 2013 BTS on-time performance data for JFK/LGA/EWR departures (336,776 flights) straight from a public Rdatasets CSV mirror at run time. Needs internet access; this is genuine historical flight data, not a toy.
- **Synthetic** — a generator (`src/data.py::make_synthetic`) built to mimic realistic patterns: bimodal departure-hour distribution peaking at 7am/5pm, delay probability pushed up during rush hours and for longer flights, weekend effect. Useful for demoing the dashboard offline, but it's a hand-tuned approximation, not real air traffic behavior.
- **Real CSV** — point `load_flights()` at your own BTS TranStats export with columns `FL_DATE, OP_CARRIER, ORIGIN, DEST, DEP_DELAY, ARR_DELAY, CRS_DEP_TIME, CRS_ARR_TIME, DISTANCE`. If the file can't be read or doesn't have enough rows, it silently falls back to synthetic data with a warning — so if your dashboard numbers look suspiciously clean, check you're not accidentally on the synthetic path.

## Feature engineering

- Departure/arrival hour gets sin/cos cyclical encoding instead of a raw 0–23 integer, so the model doesn't think 11pm and midnight are 23 hours apart.
- Carrier, origin, and destination are frequency-encoded (how common is this carrier/airport in the training data), and — this part matters — those frequencies are computed only from the first 80% of the data in chronological order, then applied to the later 20%. Otherwise you'd leak information about the test period's own traffic patterns into training.
- Distance gets bucketed into four bands, plus a weekend flag and raw month/day-of-week.
- The train/test split (`temporal_split`) is chronological, not shuffled — you train on the earlier flights and evaluate on the later ones, which is the only honest way to validate a time-ordered problem like this.

## Models

Three implementations of the same gradient boosting idea, tried in order and falling back automatically:

1. LightGBM, if installed (commented out in `requirements.txt` — install it yourself for the fastest/best results).
2. `sklearn.ensemble.GradientBoostingClassifier`, which is what actually runs by default since scikit-learn is a hard requirement here.
3. A from-scratch NumPy boosted-stump implementation (`src/core.py::GradientBoostedTrees`) that only kicks in if neither of the above is importable.

There's also a from-scratch logistic regression (gradient descent, L2 penalty, class-weighted for the delay/on-time imbalance) as a simpler baseline to compare against.

**Known limitation in the fallback path:** the from-scratch booster's tree-fitting only does a real best-split search when `max_depth<=1`. Ask it for `max_depth=5` (the default everywhere in this repo) and it silently ignores that and just splits on the first feature at its mean, every time. In practice this rarely bites anyone because scikit-learn is a required dependency and gets used first — but if you ever run this in an environment where sklearn fails to import, know that the "gradient boosting" you get isn't really building depth-5 trees. I didn't rewrite the recursive splitting logic here since that's a bigger change than a README pass should include; treat it as a documented gap rather than something silently working.

## Honest results

The `models/metrics.json` checked into this repo (from the last training run) reads:

| metric | value |
|---|---|
| accuracy | 0.74 |
| precision | 0.29 |
| recall | 0.20 |
| F1 | 0.24 |
| ROC AUC | 0.49 |
| log loss | 0.69 |
| brier score | 0.21 |

Worth being upfront about: an ROC AUC of 0.49 is indistinguishable from a coin flip, and since only about 20% of flights in that run were actually delayed, a model that just always predicts "on time" would already hit ~80% accuracy without learning anything. This particular committed run is *not* beating that trivial baseline. I'm leaving the numbers in rather than deleting them because that's the actual state of the artifact in this repo, and it's a more useful data point than a made-up "0.85 AUC" would be. If you retrain from scratch and get something better, that's genuinely dependent on which data source you point it at (synthetic vs. the real nycflights13 pull) and how much boosting you let it do.

What I'd try to actually move this: the feature set here is entirely pre-departure and mostly about *when* and *who*, not *what's happening right now*. The features that actually predict delays well in the literature — upstream aircraft rotation (is the incoming plane already late), weather at origin/destination, air traffic congestion — aren't in this dataset at all. Cyclical hour encoding and carrier frequency are a reasonable starting point but they're weak signals on their own.

## What's in the dashboard

Five tabs: a data explorer (raw table, class balance, delay rate by hour), a model lab (train GB/logistic side by side, compare ROC/confusion matrix), a feature-analysis tab (permutation importance — shuffle one column, see how much accuracy drops), a route-analysis view (delay rates by origin→destination pair, real-data only), and a cost optimizer that sweeps the decision threshold to minimize a weighted cost of missed delays vs. false alarms rather than just maximizing accuracy.

## Layout

```
src/
  core.py            LogisticRegression, GradientBoostedTrees, Standardizer, metrics from scratch
  data.py             loaders (synthetic / real CSV / live nycflights13) + feature engineering
  model.py            training entry points, evaluation, permutation importance, route analysis
  evaluate.py         metrics -> models/metrics.json
  persist.py          pickle save/load for trained models
  visualizations.py   matplotlib chart helpers used by the dashboard
train.py              CLI training pipeline
app.py                Streamlit dashboard
tests/test_smoke.py   a handful of sanity checks (shapes, cyclical encoding identity, non-crashing training)
```
