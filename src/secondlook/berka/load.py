"""Reads tables out of the Berka MariaDB dump using only the standard library (no database server needed)."""
import ast
import re
from functools import lru_cache

import pandas as pd

DEFAULT_PATH = "/home/msinghal/Documents/Hackathon/hackathon/data/berka/financial.sql"


@lru_cache(maxsize=1)
def _text(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def table(path, name):
    """Parse INSERT statements for one table out of a mysqldump file."""
    text = _text(str(path))
    m = re.search(rf"CREATE TABLE `{name}` \((.*?)\n\)", text, re.S)
    cols = re.findall(r"^\s*`(\w+)`", m.group(1), re.M)
    rows = []
    for ins in re.finditer(rf"INSERT INTO `{name}` VALUES\s*(.*?);\n", text, re.S):
        payload = ins.group(1).replace("NULL", "None")
        rows.extend(ast.literal_eval("[" + payload + "]"))
    return pd.DataFrame(rows, columns=cols)
