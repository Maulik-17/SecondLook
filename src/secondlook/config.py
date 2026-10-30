"""Single source of constants. Part 1 owns this file; changes need a message to the other two."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Real data by default; point both variables at data/fixtures/* to work before Part 1 delivers.
DATA_DIR = Path(os.environ.get("SL_DATA", ROOT / "data" / "synth"))
ART_DIR = Path(os.environ.get("SL_ART", ROOT / "artefacts"))

SEED = 20261030
N_PEOPLE = 20000
HISTORY_MONTHS = 24          # months before application
OUTCOME_MONTHS = 12          # months after the loan
THIN_FILE_MONTHS = (3, 5)    # observed window of thin-file people
SPLIT_FRAC = {"train": 0.6, "val": 0.2, "test": 0.2}   # "oot" is a separate drifted cohort

# approval policy: cost-based tau; approval-rate mode kept as an option
COST_DEFAULT = 5.0           # cost of approving someone who defaults (assumption)
COST_LOST = 1.0              # cost of rejecting someone who would repay (assumption)
TARGET_APPROVAL_RATE = None  # e.g. 0.60 to use the rate mode instead of costs

# audit / uncertainty
BOOTSTRAP_B = 1000
MIN_GROUP_N = 50
ALPHA = 0.10
EXPLORATION_EPS = 0.03
W_MAX = 50

# simulator (Part 1). Tuned so base default is about 11%, oracle AUC about 0.78, scorer AUC about 0.70 to 0.72
LTI_SCALE = 0.5          # loan instalment relative to income
POST_SHOCK_P = 0.25      # monthly chance of an unobserved expense shock after the loan
POST_SHOCK_MAX = 2.0     # shock size, multiples of monthly income (uniform 0.3 to this)
LOGIT_SLOPE = 0.55       # how strongly the simulated shortfall rate drives true risk
LOGIT_INTERCEPT = -1.0
LOGIT_FRAILTY_SD = 0.6   # unobserved heterogeneity; keeps the oracle AUC below 1
LEGACY_APPROVE_SHARE = 0.60
BIAS_FLAGS = True        # measurement bias, income-volatility gap, planted proxies (False = control run)
