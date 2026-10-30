"""Writes DATA_DIR (main + out-of-time cohort) and a sibling synth_null directory (null world, no bias)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from secondlook import config as C, contracts as K
from secondlook.sim.generate import generate


def write(tables, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if name in K.TABLES:
            K.check(name, df)
        df.to_parquet(out / f"{name}.parquet", index=False)


def main():
    main_t = generate(C.N_PEOPLE, C.SEED, "main")
    oot = generate(C.N_PEOPLE // 5, C.SEED + 1, "oot")
    off = C.N_PEOPLE
    for t in oot.values():                       # shift ids so they stay unique after concatenation
        t["id"] = t["id"] + off
    merged = {k: pd.concat([main_t[k], oot[k]], ignore_index=True) for k in main_t}
    write(merged, C.DATA_DIR)
    write(generate(C.N_PEOPLE, C.SEED + 2, "null"), C.DATA_DIR.parent / "synth_null")
    print("done", C.DATA_DIR)


if __name__ == "__main__":
    main()
