# Same Python for everyone

Target: **Python 3.14.6**, with the packages pinned in `requirements.txt` (repo root).

## Set up (each person, once)

```bash
python3.14 --version                      # must print 3.14.6; if not, install that version first (pyenv, uv, or the python.org installer)
python3.14 -m venv ~/.venv
~/.venv/bin/python -m pip install -r requirements.txt
~/.venv/bin/python -c "import numpy, scipy, pandas, pyarrow, sklearn, xgboost, statsmodels, fastapi, streamlit, plotly, pytest; print('ok')"
```
On Windows use `py -3.14 -m venv .venv` and `.venv\Scripts\python`. Put `src` on the path: `export PYTHONPATH=src`.

## Check before you start

`~/.venv/bin/python -m pip freeze | grep -i -E "numpy|pandas|scikit-learn|streamlit"` must match `requirements.txt`. If a version differs, reinstall that package from the file; do not upgrade anything.

## Rules

- Do not add or upgrade packages on your own. If you need a package that is not listed, ask the team and add it to `requirements.txt` in one commit that everyone pulls.
- `lightgbm`, `pymysql`, `sqlalchemy` are NOT part of the stack.
- `shap`, `fairlearn`, `MAPIE` are optional (comparisons only); the code must still run without them.
