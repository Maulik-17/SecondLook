"""Assumed effort weights (higher = harder to change, per standardised unit). Diagnostics, not targets."""
ACTIONABLE = ["weeks_of_buffer", "inflow_cv", "dip_months", "low_balance_days", "expense_to_income",
              "balance_level", "balance_slope", "rent_on_time", "utility_on_time", "recharge_on_time"]
ASSUMED = ["rent_on_time", "utility_on_time", "recharge_on_time"]
RATE_FEATURES = ["rent_on_time", "utility_on_time", "recharge_on_time"]
WEIGHTS = {
    "base": dict(weeks_of_buffer=2, inflow_cv=3, dip_months=3, low_balance_days=1.5, expense_to_income=2,
                 balance_level=2, balance_slope=1.5, rent_on_time=1, utility_on_time=1, recharge_on_time=0.5),
    "income_hard": dict(weeks_of_buffer=3, inflow_cv=6, dip_months=6, low_balance_days=2, expense_to_income=2,
                        balance_level=3, balance_slope=2, rent_on_time=1, utility_on_time=1, recharge_on_time=0.5),
    "behaviour_hard": dict(weeks_of_buffer=2, inflow_cv=2, dip_months=2, low_balance_days=3, expense_to_income=4,
                           balance_level=2, balance_slope=3, rent_on_time=3, utility_on_time=3, recharge_on_time=2),
}
