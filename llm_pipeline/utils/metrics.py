"""Scoring convention for the campaign: one generation per test case, no resampling.

Every test case is produced exactly once, so there is no repetition from which a
spread of R^2 could be estimated -- R^2 is a single global statistic and is reported
as a point estimate. RA is different: it is a function of the mean absolute error,
which *is* an average over test cases, so its sampling uncertainty follows from the
per-case errors themselves, with no bootstrap:

    RA = 1 - MAE / mean(y),   SE(RA) = std(|e|, ddof=1) / (sqrt(n) * mean(y))

and the reported interval is the usual normal one, RA +- z * SE(RA). The mean of the
truths is treated as fixed: it is a property of the test set, identical for every
predictor scored on it, so it carries no uncertainty relevant to comparing them.
"""
import numpy as np
from sklearn.metrics import r2_score

# Predictions above this multiple of the largest observed truth are degenerate.
# Same rule and same value as evaluate.py, so every row stays comparable.
OUTLIER_FACTOR = 3
Z95 = 1.959964


def score(y_true, y_pred, outlier_factor=OUTLIER_FACTOR, z=Z95):
    """Score one predictor on one test set. Units of y are preserved in `mae`."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = y_pred <= outlier_factor * y_true.max()
    a, b = y_true[keep], y_pred[keep]
    n = len(a)
    if n == 0:
        raise ValueError("every prediction was excluded as an outlier")
    err = np.abs(a - b)
    mae = float(err.mean())
    ybar = float(a.mean())
    ra = 1 - mae / ybar
    se = float(err.std(ddof=1)) / (np.sqrt(n) * ybar) if n > 1 else float("nan")
    half = z * se
    return dict(n=n, outliers=int((~keep).sum()), r2=float(r2_score(a, b)),
                mae=mae, ra=ra, ra_se=se, ra_ci=half,
                ra_lo=ra - half, ra_hi=ra + half)


def fmt(s, mae_scale=1.0, mae_unit=""):
    """One-line rendering: R^2 bare, RA with its 95% interval."""
    return (f"n={s['n']:5d}  R2={s['r2']:8.3f}  MAE={s['mae'] * mae_scale:9.1f}{mae_unit}  "
            f"RA={s['ra']:6.3f} [{s['ra_lo']:.3f}, {s['ra_hi']:.3f}]  "
            f"outlier={s['outliers']}")
