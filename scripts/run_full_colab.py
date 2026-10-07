import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200]
DURATIONS = ["60m", "06h", "12h", "24h"]

print("Installing Optuna...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"], check=True)

print("Downloading NOAA Atlas 14 PIDF rasters...")
for rp in RETURN_PERIODS:
    for dur in DURATIONS:
        stem = f"se{rp}yr{dur}a"
        folder = REPO_ROOT / stem
        asc = folder / f"{stem}.asc"
        if asc.exists():
            print("Already present:", asc)
            continue

        url = f"https://hdsc.nws.noaa.gov/pub/hdsc/data/se/{stem}.zip"
        zip_path = REPO_ROOT / f"{stem}.zip"
        print("Downloading:", url)
        urllib.request.urlretrieve(url, zip_path)

        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(folder)

        if not asc.exists():
            raise FileNotFoundError(f"Expected raster not found after extraction: {asc}")

rasters = list(REPO_ROOT.glob("se*yr*a/se*yr*a.asc"))
if len(rasters) != 28:
    raise RuntimeError(f"Expected 28 NOAA rasters, found {len(rasters)}")

print("NOAA rasters ready:", len(rasters))

env = os.environ.copy()
env["PIDF_SAMPLE_N"] = "250000"
env["OPTUNA_TRIALS"] = "25"
env["OUTPUT_NOTEBOOK"] = "UNG_PIDF_Model_full_colab.ipynb"

print("Building focused full experiment notebook...")
subprocess.run(
    [sys.executable, "scripts/build_ci_notebook.py"],
    check=True,
    env=env,
)

print("Executing full 250k / 25-trial experiment...")
subprocess.run(
    [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "UNG_PIDF_Model_full_colab.ipynb",
        "--output",
        "executed_UNG_PIDF_Model_full_colab.ipynb",
        "--ExecutePreprocessor.timeout=7200",
        "--ExecutePreprocessor.kernel_name=python3",
    ],
    check=True,
)

print()
print("FULL RUN COMPLETE")
print("Output: executed_UNG_PIDF_Model_full_colab.ipynb")
