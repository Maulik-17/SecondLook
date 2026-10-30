import numpy as np


def redraw(df, eps, seed=0, allow_zero=False):
    """Redraw who is observed under exploration rate eps. Returns (explored, observed, propensity) arrays.
    eps = 0 breaks positivity (rejects are never seen, weights are undefined): raises unless allow_zero=True."""
    if eps <= 0 and not allow_zero:
        raise ValueError("eps must be > 0: with no exploration the propensity of rejects is 0 and weights are undefined")
    r = np.random.default_rng(seed)
    appr = df.approved_legacy.astype(bool).values
    explored = (~appr) & (r.random(len(df)) < eps)
    observed = appr | explored
    prop = np.where(appr, 1.0, eps)
    return explored, observed, prop
