# PyInstaller spec: one-folder build. Run from the repository root:  pyinstaller packaging/secondlook.spec
# app/, src/, data/, artefacts/ and .streamlit/ are copied next to the executable by the build workflow.
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

datas, binaries, hiddenimports = [], [], []
for pkg in ["streamlit", "plotly", "sklearn", "scipy", "statsmodels", "xgboost", "pyarrow", "pandas", "numpy",
            "fastapi", "starlette", "pydantic", "uvicorn", "joblib", "altair", "narwhals"]:
    datas += collect_data_files(pkg)
    binaries += collect_dynamic_libs(pkg)
    hiddenimports += collect_submodules(pkg, filter=lambda n: ".test" not in n and not n.endswith(".conftest"))
for pkg in ["streamlit", "fastapi", "starlette", "pydantic", "uvicorn", "xgboost", "plotly", "pandas", "numpy", "scipy"]:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass
hiddenimports += collect_submodules("secondlook")

a = Analysis(["launcher.py"], pathex=[os.path.join(SPECPATH, "..", "src")], datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             excludes=["shap", "fairlearn", "mapie", "matplotlib", "pytest", "tkinter", "IPython", "torch", "tensorflow", "jax",
                       "cv2", "PyQt5", "PyQt6", "PySide6", "numba", "llvmlite", "triton", "av", "yt_dlp", "sympy"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SecondLook", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="SecondLook")
