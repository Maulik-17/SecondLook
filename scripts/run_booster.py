"""Optional: compare the logistic scorecard with a monotone XGBoost booster. Writes <SL_ART>/booster_comparison.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss

from secondlook import config, contracts as K
from secondlook.model.booster import Booster
from secondlook.model.scorer import Scorer
from train import auc_ci, ece

DISCLAIMER = ("Synthetic data; an audit-tool demonstration, not a lending decision tool. "
              "Results come from one simulated dataset and one seed; they do not show how either model would behave on real applicants.")


def group_gaps(df, decision):
    d = df.assign(a=decision)
    out = {}
    for g in K.GROUPS:
        r = d.groupby(g).a.agg(["mean", "size"])
        r = r[r["size"] >= config.MIN_GROUP_N]
        out[g] = {"rates": {str(k): float(v) for k, v in r["mean"].items()}, "gap": float(r["mean"].max() - r["mean"].min())}
    return out


def eval_model(m, sp, y):
    res = {}
    for name in ["test", "oot"]:
        p = m.predict_proba(sp[name])
        dec = m.decide(sp[name])
        res[name] = {"auc": auc_ci(y[name], p), "brier": float(brier_score_loss(y[name], p)), "ece": ece(p, y[name]),
                     "approval_rate": float(dec.mean()), "group_approval": group_gaps(sp[name], dec)}
    return res


def stability(imps, k=5):
    """Mean pairwise Spearman correlation of importances, and mean share of top-k features shared between resamples."""
    top = lambda v: set(np.argsort(-v)[:k])
    rho, ov = [], []
    for i in range(len(imps)):
        for j in range(i + 1, len(imps)):
            rho.append(spearmanr(imps[i], imps[j])[0])
            ov.append(len(top(imps[i]) & top(imps[j])) / k)
    return {"mean_rank_correlation": float(np.mean(rho)), "mean_top5_overlap": float(np.mean(ov)), "n_pairs": len(rho)}


def main(n_boot=10):
    D, A = config.DATA_DIR, config.ART_DIR
    feats = json.loads((A / "metrics.json").read_text())["kept_features"]
    df = pd.read_parquet(D / "people.parquet").merge(pd.read_parquet(D / "features.parquet"), on="id")
    sp = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test", "oot"]}
    y = {s: sp[s].y_true.values for s in sp}

    lr = Scorer(feats).fit(sp["train"], y["train"], sp["val"], y["val"])
    lr.set_tau_cost(sp["val"], y["val"])
    bo = Booster(feats).fit(sp["train"], y["train"], sp["val"], y["val"])
    bo.set_tau_cost(sp["val"], y["val"])

    rng = np.random.default_rng(config.SEED)
    shap_imps, lr_imps, n = [], [], len(sp["train"])
    for _ in range(n_boot):
        tr = sp["train"].iloc[rng.integers(0, n, n)].reset_index(drop=True)
        b = Booster(feats).fit(tr, tr.y_true.values, sp["val"], y["val"])
        shap_imps.append(b.shap_importance(sp["test"]))
        s = Scorer(feats).fit(tr, tr.y_true.values, sp["val"], y["val"])
        lr_imps.append(np.abs(s.lr_.coef_[0]))        # coefficients on standardised features, comparable across features

    out = {"disclaimer": DISCLAIMER, "kept_features": feats, "n_bootstrap_refits": n_boot,
           "tau": {"logistic": lr.tau, "booster": bo.tau},
           "logistic": eval_model(lr, sp, y), "booster": eval_model(bo, sp, y),
           "stability": {"booster_shap": stability(shap_imps), "logistic_coefficients": stability(lr_imps)},
           "booster_mean_abs_shap_full_fit": dict(zip(feats, map(float, bo.shap_importance(sp["test"]))))}
    (A / "booster_comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["tau", "stability"]}, indent=1))
    for m in ["logistic", "booster"]:
        for s in ["test", "oot"]:
            r = out[m][s]
            print(m, s, "auc", [round(x, 3) for x in r["auc"]], "brier", round(r["brier"], 4), "ece", round(r["ece"], 4),
                  "appr", round(r["approval_rate"], 3), {g: round(v["gap"], 3) for g, v in r["group_approval"].items()})


if __name__ == "__main__":
    main()
