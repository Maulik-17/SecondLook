"""Shared schemas. The three parts exchange ONLY these files and these function signatures (docs/SPLIT_CONTRACT.md)."""
import pandas as pd

GROUPS = {
    "gender": ["F", "M"],
    "urban_rural": ["urban", "rural"],
    "region": ["R1", "R2", "R3", "R4", "R5", "R6"],
    "occupation": ["gig_driver", "gig_delivery", "shopkeeper", "student_parttime", "salaried_informal"],
    "thin_file": [0, 1],
    "shared_phone": [0, 1],
    "document_poor": [0, 1],
}
# never given to the model; kept for audit only
PROTECTED_FOR_AUDIT = list(GROUPS)

FEATURES = [
    "mean_inflow", "median_inflow", "inflow_cv", "dip_months", "spike_months",
    "weeks_of_buffer", "low_balance_days", "expense_to_income", "balance_level",
    "balance_slope", "rent_on_time", "utility_on_time", "recharge_on_time",
    "months_observed", "emi_to_income",
    "night_active_share", "local_market_share",   # planted proxies (gender, region): no effect on true risk
]
# monotone expert direction: +1 raises risk, -1 lowers risk, 0 unconstrained
FEATURE_DIRECTION = {
    "mean_inflow": -1, "median_inflow": -1, "inflow_cv": +1, "dip_months": +1,
    "spike_months": 0, "weeks_of_buffer": -1, "low_balance_days": +1,
    "expense_to_income": +1, "balance_level": -1, "balance_slope": -1,
    "rent_on_time": -1, "utility_on_time": -1, "recharge_on_time": -1,
    "months_observed": -1, "emi_to_income": +1,
    "night_active_share": 0, "local_market_share": 0,
}
# source label shown in the data dictionary
FEATURE_SOURCE = {f: "assumed" if f.endswith("_on_time") else "literature-supported" for f in FEATURES}
FEATURE_SOURCE.update({"night_active_share": "assumed", "local_market_share": "assumed"})

TABLES = {  # file name (parquet) -> required columns
    "people": ["id", "split", *GROUPS, "loan_amount", "loan_term", "y_true", "r_true"],
    "features": ["id", *FEATURES],
    "monthly": ["id", "month", "inflow", "outflow", "balance"],        # month 1..24 history, 25..36 outcome
    "scores": ["id", "p_default", "logit", "decision", "tau"],         # every person, calibrated
    "lender_history": ["id", "approved_legacy", "explored", "propensity", "observed", "y_obs"],
}
LATENTS_COLUMNS = ["id", "mu_income", "sigma_income", "shock_prob", "rho_persist", "buffer0"]  # latents.parquet, Part 1 only


def check(name: str, df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in TABLES[name] if c not in df.columns]
    assert not missing, f"{name}: missing columns {missing}"
    assert df["id"].is_unique or name == "monthly", f"{name}: id not unique"
    return df
