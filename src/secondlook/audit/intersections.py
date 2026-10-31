"""Intersection scan: burden on the creditworthy rejected and approval rate for pairs of group attributes.
Same burden definition as burden.burden_rows (finite effort summed over rejected-and-creditworthy / group size).
Cells under config.MIN_GROUP_N are skipped. Many cells are examined, so some will look extreme by chance."""
from itertools import combinations

import numpy as np

from .. import config as C
from ..contracts import GROUPS
from .metrics import boot_mean, wilson, _py


def scan(te, eff, rg, rejected, attrs=None, B=100, top=10):
    attrs = list(attrs or GROUPS)
    finite = np.where(np.isfinite(eff), eff, 0.0)
    per_all = finite * rg
    rows, skipped = [], 0
    for a1, a2 in combinations(attrs, 2):
        for l1 in GROUPS[a1]:
            for l2 in GROUPS[a2]:
                m = ((te[a1] == l1) & (te[a2] == l2)).values
                n = int(m.sum())
                if n < C.MIN_GROUP_N:
                    skipped += 1
                    continue
                rows.append({"a1": a1, "l1": _py(l1), "a2": a2, "l2": _py(l2), "n": n,
                             "n_rejected_good": int(rg[m].sum()),
                             "burden": boot_mean(per_all[m], B=B) if B else [float(per_all[m].mean())] * 3,
                             "approval": wilson(int((~rejected[m]).sum()), n)})
    worst = sorted(rows, key=lambda r: -r["burden"][1])[:top]
    overall = float(per_all.mean()) if len(per_all) else float("nan")
    return {"rows": rows, "worst": worst, "n_cells_examined": len(rows), "n_cells_skipped_small": skipped,
            "overall_burden": overall, "min_n": C.MIN_GROUP_N,
            "caution": (f"{len(rows)} intersections were examined (skipped {skipped} with n < {C.MIN_GROUP_N}). "
                        f"With 95% intervals about {0.05 * len(rows):.0f} would look unusual by chance alone, "
                        "so treat the worst cells as leads to check, not findings. No multiple-comparison correction is applied.")}
