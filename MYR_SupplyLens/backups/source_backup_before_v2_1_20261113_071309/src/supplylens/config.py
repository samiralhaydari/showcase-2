from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
SCALER_DIR = PROJECT_ROOT / "artifacts" / "scalers"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
FORECAST_DIR = PROJECT_ROOT / "artifacts" / "forecasts"
DB_PATH = PROJECT_ROOT / "data" / "supplylens.sqlite"

DATA_URL = "https://storage.data.gov.my/finsector/exr/daily_1200.csv"
DATASET_PAGE = "https://data.gov.my/data-catalogue/exchangerates_daily_1200"

PRIORITY_CURRENCIES = ["USD", "CNY", "IDR", "SGD", "THB", "JPY", "KRW"]
CURRENCY_NAMES = {
    "USD": "US Dollar", "CNY": "Chinese Yuan", "IDR": "Indonesian Rupiah",
    "SGD": "Singapore Dollar", "THB": "Thai Baht", "JPY": "Japanese Yen",
    "KRW": "South Korean Won", "PHP": "Philippine Peso", "VND": "Vietnamese Dong",
    "BND": "Brunei Dollar", "EUR": "Euro", "GBP": "British Pound",
    "AUD": "Australian Dollar", "CAD": "Canadian Dollar", "CHF": "Swiss Franc",
    "HKD": "Hong Kong Dollar", "TWD": "Taiwan Dollar", "INR": "Indian Rupee",
    "NZD": "New Zealand Dollar"
}

MODEL_VERSION = "v2"
LOOKBACK = 90
FORECAST_HORIZON = 63
DISPLAY_HORIZONS = [5, 21, 42, 63]
# A common 63-session origin spacing makes all reported horizons non-overlapping.
WALK_FORWARD_SPACING = FORECAST_HORIZON
CALIBRATION_MIN_SAMPLES = 30
ARIMA_MAX_HISTORY = 756
FEATURE_COLUMNS = [
    "log_rate",
    "return_1d",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "volatility_5",
    "volatility_20",
    "volatility_60",
    "range_20",
]
