import pandas as pd

from .. import config as C
from ..contracts import GROUPS

DISCLAIMER = "Synthetic data; an audit-tool demonstration, not a finding about real lenders and not a lending decision tool."


def load(data_dir=None):
    d = data_dir or C.DATA_DIR
    ppl = pd.read_parquet(d / "people.parquet")
    F = pd.read_parquet(d / "features.parquet")
    S = pd.read_parquet(d / "scores.parquet")
    LH = pd.read_parquet(d / "lender_history.parquet")
    S = S.drop(columns=[c for c in ("tau",) if c in S.columns])
    return ppl.merge(F, on="id").merge(S, on="id").merge(LH, on="id")


def levels():
    return [(a, l) for a, ls in GROUPS.items() for l in ls]


def mask(df, attr, level):
    return df[attr] == level


def test_pop(df):
    return df[df.split == "test"].reset_index(drop=True)
