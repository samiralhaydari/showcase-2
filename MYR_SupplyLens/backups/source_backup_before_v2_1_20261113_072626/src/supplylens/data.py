import requests
import pandas as pd

from .config import DATA_URL, DATA_RAW_DIR, PRIORITY_CURRENCIES


def download_raw_data(force_download=False):
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW_DIR / "daily_1200.csv"
    if force_download or not path.exists():
        response = requests.get(DATA_URL, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def load_exchange_rates(force_download=False):
    # Returns daily middle rates as MYR per one unit of foreign currency.
    path = download_raw_data(force_download)
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    if "date" not in raw.columns:
        raise ValueError(f"Expected date column. Found: {list(raw.columns)}")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])

    currency_columns = [
        c for c in raw.columns if len(c) == 3 and c.isalpha()
    ]
    if not currency_columns:
        currency_columns = [c for c in PRIORITY_CURRENCIES if c in raw.columns]
    if not currency_columns:
        raise ValueError("No three-letter currency columns found.")

    if "rate_type" in raw.columns:
        rate_type = raw["rate_type"].astype(str).str.strip().str.lower()
        middle = raw.loc[rate_type.eq("middle")].copy()
        if not middle.empty:
            raw = middle
        else:
            bs = raw.loc[rate_type.isin(["buying", "selling"])].copy()
            if bs.empty:
                raise ValueError("No middle or buying/selling rate rows found.")
            raw = bs.groupby("date", as_index=False)[currency_columns].mean()

    for col in currency_columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    rates = (
        raw[["date"] + currency_columns]
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .dropna(how="all")
    )
    rates = rates.loc[:, rates.notna().any()]
    rates.columns = [str(c).upper() for c in rates.columns]
    return rates


def get_currency_series(rates, currency):
    currency = currency.upper()
    if currency not in rates.columns:
        raise ValueError(
            f"{currency} is unavailable. Available: {', '.join(rates.columns)}"
        )
    series = pd.to_numeric(rates[currency], errors="coerce").dropna()
    series = series[series > 0].sort_index()
    series.name = currency
    return series


def available_priority_currencies(rates):
    return [c for c in PRIORITY_CURRENCIES if c in rates.columns]
