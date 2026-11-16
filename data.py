"""ERCOT summer data: hourly COAST load and HB_HOUSTON real-time prices.

Every summer (June-August) is returned as a DataFrame indexed by the
hour-ending timestamp, with columns ``load`` (MW, COAST zone) and
``price`` ($/MWh, hourly mean of the four 15-minute intervals).
Hour ending 01:00 on 1 June is the first hour and hour ending 00:00 on
1 September the last, giving 92 * 24 = 2,208 hours.

Quirks handled (design spec, Section 2):
* 2015-2016 load: real datetimes with millisecond offsets -> rounded.
* 2017+ load: text 'MM/DD/YYYY HH:MM', '24:00' = midnight of next day,
  ' DST' suffix on the repeated November hour; 2022 has one real
  datetime cell.
* 2015-2016 price sheets are split and suffixed (Jun_9, Jul_10, ...).
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

# Folder with the ERCOT files (see README); default: the parent of this folder.
ROOT = os.environ.get("RESS_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

LOAD_FILES = {
    2015: "native_load_2015.xls",
    2016: "native_Load_2016.xlsx",
    2017: "native_Load_2017.xlsx",
    2022: "ercot_recent/Native_Load_2022.xlsx",
    2023: "ercot_recent/Native_Load_2023.xlsx",
    2024: "ercot_recent/Native_Load_2024.xlsx",
    2025: "ercot_recent/Native_Load_2025.xlsx",
}
PRICE_TMPL = "rpt.00013061.0000000000000000.RTMLZHBSPP_{y}.xlsx"

STORM = (pd.Timestamp("2024-07-06"), pd.Timestamp("2024-07-16"))  # delivery dates, inclusive

EVAL_SUMMERS = [2016, 2017, 2023, 2024, 2025]
CALIB_OF = {2016: 2015, 2017: 2016, 2023: 2022, 2024: 2023, 2025: 2024}


def _summer_index(year: int, first_month: int = 6) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-{first_month:02d}-01 01:00", f"{year}-09-01 00:00", freq="h")


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _parse_hour_ending(v) -> pd.Timestamp:
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"):
        return pd.Timestamp(v).round("h")
    s = str(v).replace(" DST", "").strip()
    date, hm = s.split(" ")
    hh, mm = hm.split(":")
    return pd.Timestamp(pd.to_datetime(date, format="%m/%d/%Y")) + pd.Timedelta(hours=int(hh), minutes=int(mm))


def load_series(year: int, first_month: int = 6) -> pd.Series:
    df = pd.read_excel(os.path.join(ROOT, LOAD_FILES[year]))
    tcol = "Hour_End" if "Hour_End" in df.columns else "Hour Ending"
    df = df[[tcol, "COAST"]].dropna()
    ts = df[tcol].map(_parse_hour_ending)
    s = pd.Series(df["COAST"].astype(float).values, index=ts)
    s = s[~s.index.duplicated(keep="first")]  # repeated DST hour (Nov only)
    idx = _summer_index(year, first_month)
    return s.reindex(idx)


def price_series(year: int, first_month: int = 6) -> pd.Series:
    folder = ROOT if year < 2020 else os.path.join(ROOT, "ercot_recent")
    xl = pd.ExcelFile(os.path.join(folder, PRICE_TMPL.format(y=year)))
    keep = MONTHS[first_month - 1:8]
    sheets = [sh for sh in xl.sheet_names if sh.split("_")[0] in keep]
    df = pd.concat([pd.read_excel(xl, sheet_name=sh) for sh in sheets], ignore_index=True)
    df = df[df["Settlement Point Name"] == "HB_HOUSTON"]
    df = df[df["Repeated Hour Flag"].astype(str).str.upper().str[0] != "Y"]
    g = df.groupby(["Delivery Date", "Delivery Hour"])["Settlement Point Price"].mean().reset_index()
    ts = pd.to_datetime(g["Delivery Date"], format="%m/%d/%Y") + pd.to_timedelta(g["Delivery Hour"], unit="h")
    s = pd.Series(g["Settlement Point Price"].values, index=ts).sort_index()
    return s.reindex(_summer_index(year, first_month))


def season(year: int) -> pd.DataFrame:
    """1 April - 31 August (hour ending), cached. April-May serve only as
    look-back for the event rule and the system-load filter."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"season_{year}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    df = pd.DataFrame({"load": load_series(year, 4), "price": price_series(year, 4)})
    if df.isna().any().any():
        raise ValueError(f"{year}: missing values\n{df[df.isna().any(axis=1)]}")
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


def summer(year: int) -> pd.DataFrame:
    """Load + price for one summer (June-August)."""
    s = season(year)
    return s.loc[_summer_index(year)]


def storm_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True for hours whose delivery date lies in the excluded storm period."""
    d = (index - pd.Timedelta(hours=1)).normalize()
    return np.asarray((d >= STORM[0]) & (d <= STORM[1]))


def summary_table() -> pd.DataFrame:
    rows = []
    for y in sorted(LOAD_FILES):
        df = summer(y)
        rows.append(dict(summer=y, load_mean=df.load.mean(), load_peak=df.load.max(),
                         price_median=df.price.median(), price_p99=df.price.quantile(0.99),
                         hours_gt100=int((df.price > 100).sum()), n=len(df)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    print(summary_table().round(1).to_string(index=False))
