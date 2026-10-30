import numpy as np

SKIP = {"night_active_share", "local_market_share"}   # planted proxies are never shown as reasons

_PAIRS = {
    "mean_inflow": ("Your average monthly income is low", "Your average monthly income is high"),
    "median_inflow": ("Your typical monthly income is low", "Your typical monthly income is high"),
    "inflow_cv": ("Your monthly income varies a lot", "Your monthly income is stable"),
    "dip_months": ("Your income dipped well below its usual level in several recent months",
                   "Your income stayed steady month to month"),
    "spike_months": ("Your income had several unusually high months, so it is less predictable",
                     "Your income had few unusually high months"),
    "weeks_of_buffer": ("Your savings cover only a few weeks of spending", "Your savings cover many weeks of spending"),
    "low_balance_days": ("Your balance was low on many days", "Your balance rarely ran low"),
    "expense_to_income": ("You spend most of what you earn", "You spend well under what you earn"),
    "balance_level": ("Your balance is small compared with your monthly income",
                      "Your balance is large compared with your monthly income"),
    "balance_slope": ("Your balance has been falling over the recent months", "Your balance has been rising over the recent months"),
    "rent_on_time": ("Rent payments were late more often (assumed signal)", "Rent payments were on time (assumed signal)"),
    "utility_on_time": ("Utility bills were paid late more often (assumed signal)", "Utility bills were paid on time (assumed signal)"),
    "recharge_on_time": ("Phone recharges were late more often (assumed signal)", "Phone recharges were on time (assumed signal)"),
    "months_observed": ("Only a short account history is available", "A longer account history is available"),
    "emi_to_income": ("The instalment is large compared with your income", "The instalment is small compared with your income"),
}
TEXT = {}
for _f, (_up, _down) in _PAIRS.items():
    TEXT[(_f, "raises risk")] = _up
    TEXT[(_f, "lowers risk")] = _down


def reasons(scorer, x_row, top=5):
    """x_row: one-row DataFrame. Exact for the logistic scorecard: contribution_j = w_j * (x_j - mean_j)."""
    lp = scorer.linear_params()
    mean = dict(zip(scorer.features, scorer.mean_))
    out = []
    for f, w in zip(lp["features"], lp["w"]):
        if f in SKIP:
            continue
        c = float(w * (float(x_row[f].iloc[0]) - mean[f]))
        d = "raises risk" if c > 0 else "lowers risk"
        out.append({"feature": f, "contribution": c, "direction": d, "text": TEXT[(f, d)]})
    return sorted(out, key=lambda r: -abs(r["contribution"]))[:top]
