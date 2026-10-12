"""Project constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_FILE = DATA_DIR / "download_traces.csv.gz"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# Raca et al., "Beyond Throughput, The Next Generation: A 5G Dataset with Channel and
# Context Metrics", ACM MMSys 2020. The zip lives in the git history of the repository.
DATA_COMMIT = "3175653a6fc09611b7dba15e0e128183e6af7c6c"
DATA_URL = (f"https://raw.githubusercontent.com/uccmisl/5Gdataset/{DATA_COMMIT}/"
            "5G-production-dataset.zip")

RANDOM_STATE = 42
HORIZONS = [1, 3, 5]          # forecast horizons in seconds
MAX_GAP_FILL = 3              # seconds of missing samples filled forward inside a trace

# Carrier frequencies used for the Doppler feature (assumption, see paper):
# 5G NR n78 (3.5 GHz) and LTE band 3 (1.8 GHz) are the bands deployed in Ireland in 2020.
CARRIER_HZ = {"5G": 3.5e9, "LTE": 1.8e9, "3G": 2.1e9}
SPEED_OF_LIGHT = 299_792_458.0

# Spatial grid used by the space-time prior (~110 m x 110 m cells at 52 deg N)
GRID_LAT = 0.001
GRID_LON = 0.0016
