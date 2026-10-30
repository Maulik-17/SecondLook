NOTE = ("When base rates differ, equal calibration, equal TPR and equal FPR cannot all hold "
        "(Kleinberg et al.; Chouldechova). Effort and burden figures are diagnostics, not targets.")


def tradeoff(rows):
    """rows: group_table output. Positive = repays, predicted positive = approved.
    Implied FPR (approved among defaulters) = p/(1-p) * (1-PPV)/PPV * TPR with p the prevalence of repayers."""
    base, detail = {}, []
    for r in rows:
        key = f"{r['attribute']}:{r['level']}"
        br = r["base_rate"][1]
        base[key] = br
        p = 1 - br
        ppv, tpr = r["ppv"][1], r["tpr"][1]
        try:
            implied = p / (1 - p) * (1 - ppv) / ppv * tpr
        except ZeroDivisionError:
            implied = float("nan")
        detail.append({"attribute": r["attribute"], "level": r["level"], "n": r["n"], "base_rate": br,
                       "ppv": ppv, "tpr": tpr, "fpr_measured": r["fpr"][1], "fpr_implied": float(implied)})
    return {"base_rates": base, "note": NOTE, "rows": detail}
