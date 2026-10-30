"""Writes small random data in the exact contract schema so Parts 2 and 3 can build before Part 1 delivers.
Usage:  ~/.venv/bin/python scripts/make_fixtures.py
        export SL_DATA=data/fixtures/synth SL_ART=data/fixtures/artefacts
The numbers are NOT meaningful; only the schema is."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secondlook import config, contracts as C
from secondlook.model.scorer import Scorer, save_model

N, rng = 3000, np.random.default_rng(1)
out, art = config.ROOT / "data/fixtures/synth", config.ROOT / "data/fixtures/artefacts"
out.mkdir(parents=True, exist_ok=True); art.mkdir(parents=True, exist_ok=True)

ppl = pd.DataFrame({"id": np.arange(N)})
for g, levels in C.GROUPS.items():
    ppl[g] = rng.choice(levels, N)
ppl["thin_file"] = (rng.random(N) < 0.4).astype(int)
ppl["split"] = rng.choice(["train", "val", "test"], N, p=[.6, .2, .2])
ppl.loc[rng.random(N) < 0.1, "split"] = "oot"
ppl["loan_amount"] = rng.integers(10, 100, N) * 1000
ppl["loan_term"] = rng.choice([6, 12, 18, 24], N)
risk = rng.normal(size=N)
ppl["r_true"] = 1 / (1 + np.exp(-(-2.1 + 0.9 * risk)))
ppl["y_true"] = (rng.random(N) < ppl.r_true).astype(int)

F = pd.DataFrame({"id": np.arange(N)})
for f in C.FEATURES:
    F[f] = rng.normal(size=N) + 0.5 * C.FEATURE_DIRECTION[f] * risk
F["months_observed"] = np.where(ppl.thin_file == 1, rng.integers(3, 6, N), 24)
mon = pd.DataFrame({"id": np.repeat(np.arange(N), 36), "month": np.tile(np.arange(1, 37), N)})
mon["inflow"] = np.abs(rng.normal(20000, 5000, len(mon)))
mon["outflow"] = mon.inflow * rng.uniform(.7, 1.0, len(mon))
mon["balance"] = mon.groupby("id").apply(lambda d: (d.inflow - d.outflow).cumsum() + 5000).explode().astype(float).values

tr, va = ppl.split == "train", ppl.split == "val"
s = Scorer().fit(F[tr], ppl.y_true[tr], F[va], ppl.y_true[va])
s.set_tau_cost(F[va], ppl.y_true[va].values)
save_model(s, art / "model.joblib")
sc = pd.DataFrame({"id": F.id, "p_default": s.predict_proba(F), "logit": s.logit(F), "tau": s.tau})
sc["decision"] = (sc.p_default <= s.tau).astype(int)

appr = (rng.random(N) < np.clip(1 - ppl.r_true * 3, .05, .95))
expl = rng.random(N) < config.EXPLORATION_EPS
obs = appr | expl
lh = pd.DataFrame({"id": F.id, "approved_legacy": appr.astype(int), "explored": (expl & ~appr).astype(int)})
lh["propensity"] = np.where(appr, 1.0, config.EXPLORATION_EPS)  # e(x)
lh["observed"] = obs.astype(int); lh["y_obs"] = np.where(obs, ppl.y_true, np.nan)

for name, df in {"people": ppl, "features": F, "monthly": mon, "scores": sc, "lender_history": lh}.items():
    C.check(name, df).to_parquet(out / f"{name}.parquet", index=False)
print("fixtures written to", out, "| tau =", round(s.tau, 3))
