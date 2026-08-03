"""
data_prep.py

Reads the raw Bloomberg export of NASDAQ-100 (NDX) weekly price and trailing
12-month dividend yield data, cleans it, and writes a tidy CSV used by every
downstream script.

Input : data/raw/NDX_Weekly.csv
Output: data/processed/ndx_weekly_pd_ratio.csv

Output columns:
    date              weekly observation date
    price             NDX index level (PX_LAST)
    div_yield_12m     trailing 12-month dividend yield, gross (%) (EQY_DVD_YLD_12M)
    pd_ratio          price-to-dividend ratio (as exported by Bloomberg)
    log_pd_ratio      natural log of pd_ratio -- the series tested for bubbles
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/NDX_Weekly.csv"
OUT_PATH = "data/processed/ndx_weekly_pd_ratio.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    # Bloomberg export: 5 metadata/header rows, then "date,price,yield,pd_ratio" rows,
    # followed by a block of trailing blank rows. utf-8-sig strips the BOM.
    df = pd.read_csv(
        path,
        skiprows=5,
        header=None,
        names=["date", "price", "div_yield_12m", "pd_ratio"],
        encoding="utf-8-sig",
    )
    df = df.dropna(subset=["date", "price"])
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    for col in ["price", "div_yield_12m", "pd_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price", "pd_ratio"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_clean_dataset(path: str = RAW_PATH) -> pd.DataFrame:
    df = load_raw(path)
    df["log_pd_ratio"] = np.log(df["pd_ratio"])
    return df


def main():
    df = build_clean_dataset()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} weekly observations to {OUT_PATH}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()
