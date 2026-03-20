import argparse
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

# =========================
# CONFIG
# =========================

JHU_REPO_API = "https://api.github.com/repos/CSSEGISandData/measles_data/contents"
JHU_RAW_BASE = "https://raw.githubusercontent.com/CSSEGISandData/measles_data/main"
DEFAULT_HISTORY_LOCAL = "measles_county_all_updates.csv"

# Local supporting files you already downloaded
LOCAL_COMMUTER_CSV = "./cache/commuter.csv"
LOCAL_VACC_CSV = "./cache/Vaccination_Coverage_and_Exemptions_among_Kindergartners_20260320.csv"
LOCAL_GAZETTEER_TXT = "./cache/2025_Gaz_counties_national.txt"
LOCAL_COUNTY_GEOJSON = "./cache/geojson-counties-fips.json"

USER_AGENT = "measles-forecast-app/1.2"

STATE_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR"
}

AIR_TRAVEL_PROXY = {
    "CA": 1.35, "TX": 1.20, "FL": 1.18, "NY": 1.30, "IL": 1.12, "GA": 1.15,
    "NJ": 1.10, "WA": 1.08, "MA": 1.05, "VA": 1.00, "MD": 0.95, "DC": 1.10,
}

STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR"
}


# =========================
# HELPERS
# =========================

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def standardize_fips(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"\D", "", s)
    if not s:
        return None
    return s.zfill(5)


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def state_abbr_from_any(val: str) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return STATE_NAME_TO_ABBR.get(s.lower(), "")


def split_county_and_state(location_name: str) -> Tuple[str, str]:
    if pd.isna(location_name):
        return "", ""
    s = str(location_name).strip()

    if "," in s:
        left, right = s.rsplit(",", 1)
        county = left.strip()
        state_part = right.strip()
        state_abbr = state_abbr_from_any(state_part) or (state_part.upper() if len(state_part) == 2 else "")
        return county, state_abbr

    return s, ""


def read_csv_flexible(path: str, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not read CSV {path}: {last_err}")


# =========================
# STEP 1: HISTORY INPUT
# =========================

def discover_latest_repo_csv() -> str:
    url = JHU_REPO_API
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    resp.raise_for_status()
    items = resp.json()

    candidates = []
    for item in items:
        name = item.get("name", "")
        if name.lower().endswith(".csv") and "county" in name.lower():
            candidates.append(name)

    if not candidates:
        return f"{JHU_RAW_BASE}/{DEFAULT_HISTORY_LOCAL}"

    candidates.sort()
    latest = candidates[-1]
    return f"{JHU_RAW_BASE}/{latest}"


def load_history_csv(history_csv: Optional[str]) -> pd.DataFrame:
    if history_csv and os.path.exists(history_csv):
        return read_csv_flexible(history_csv)

    url = discover_latest_repo_csv()
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize measles history into:
    date, county_fips, county, state, reported_cases

    Important:
    value is treated as reported/incremental cases.
    Weekly cumulative is built later from weekly sums.
    """
    df = df.copy()
    original_cols = list(df.columns)
    cols_lower = {c.lower().strip(): c for c in df.columns}

    required_schema = {
        "location_name", "location_id", "location_type", "date", "outcome_type", "value"
    }

    if required_schema.issubset(set(cols_lower.keys())):
        location_name_col = cols_lower["location_name"]
        location_id_col = cols_lower["location_id"]
        location_type_col = cols_lower["location_type"]
        date_col = cols_lower["date"]
        outcome_type_col = cols_lower["outcome_type"]
        value_col = cols_lower["value"]

        temp = df.copy()
        temp[location_type_col] = temp[location_type_col].astype(str).str.strip().str.lower()
        temp[outcome_type_col] = temp[outcome_type_col].astype(str).str.strip().str.lower()

        temp = temp[temp[location_type_col] == "county"].copy()

        if temp.empty:
            raise ValueError(
                "No county rows found after filtering location_type == 'county'. "
                f"Available location_type values: {sorted(df[location_type_col].dropna().astype(str).unique().tolist())}"
            )

        keep_mask = (
            temp[outcome_type_col].str.contains("case", na=False) |
            temp[outcome_type_col].str.contains("confirmed", na=False) |
            temp[outcome_type_col].str.contains("total", na=False) |
            temp[outcome_type_col].str.contains("cumulative", na=False)
        )
        filtered = temp[keep_mask].copy()
        if filtered.empty:
            filtered = temp.copy()

        filtered = filtered.sort_values([location_id_col, date_col])

        out = pd.DataFrame({
            "date": pd.to_datetime(filtered[date_col], errors="coerce"),
            "county_fips": filtered[location_id_col].apply(standardize_fips),
            "county_raw": filtered[location_name_col].astype(str).str.strip(),
            "reported_cases": pd.to_numeric(filtered[value_col], errors="coerce"),
        })

        out = out.dropna(subset=["date", "county_fips", "reported_cases"]).copy()
        out = out[out["county_fips"].str.match(r"^\d{5}$", na=False)].copy()
        out = out[out["county_fips"] != "00000"].copy()

        county_state = out["county_raw"].apply(split_county_and_state)
        out["county"] = county_state.apply(lambda x: x[0])
        out["state"] = county_state.apply(lambda x: x[1])

        out["reported_cases"] = out["reported_cases"].clip(lower=0)

        out = (
            out.groupby(["county_fips", "date"], as_index=False)
               .agg({
                   "county": "last",
                   "state": "last",
                   "reported_cases": "sum"
               })
        )

        return out.sort_values(["county_fips", "date"]).reset_index(drop=True)

    def find_col(possibles):
        for p in possibles:
            for c in df.columns:
                if c.lower().strip() == p.lower().strip():
                    return c
        for p in possibles:
            for c in df.columns:
                if p.lower() in c.lower():
                    return c
        return None

    date_col = find_col(["date", "report_date", "updatedate", "update_date"])
    fips_col = find_col(["fips", "county_fips", "fips_code", "geoid", "location_id"])
    county_col = find_col(["county", "county_name", "admin2", "location_name"])
    state_col = find_col(["state", "province_state", "province_state_name"])
    cases_col = find_col(["cases", "new_cases", "reported_cases", "value"])

    if date_col is None:
        raise ValueError(f"Could not find a date column. Columns: {original_cols}")
    if fips_col is None:
        raise ValueError(f"Could not find a county FIPS column. Columns: {original_cols}")
    if cases_col is None:
        raise ValueError(f"Could not find a case count column. Columns: {original_cols}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "county_fips": df[fips_col].apply(standardize_fips),
        "county": df[county_col] if county_col else "",
        "state": df[state_col] if state_col else "",
        "reported_cases": pd.to_numeric(df[cases_col], errors="coerce"),
    })

    out = out.dropna(subset=["date", "county_fips", "reported_cases"]).copy()
    out = out[out["county_fips"].str.match(r"^\d{5}$", na=False)].copy()
    out = out[out["county_fips"] != "00000"].copy()
    out["county"] = out["county"].fillna("").astype(str)
    out["state"] = out["state"].fillna("").astype(str).apply(state_abbr_from_any)
    out["reported_cases"] = out["reported_cases"].clip(lower=0)

    out = (
        out.groupby(["county_fips", "date"], as_index=False)
           .agg({
               "county": "last",
               "state": "last",
               "reported_cases": "sum"
           })
    )

    return out.sort_values(["county_fips", "date"]).reset_index(drop=True)


def build_weekly_history(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df["week"] = df["date"].dt.to_period("W-SAT").dt.start_time

    weekly = (
        df.groupby(["county_fips", "week"], as_index=False)
          .agg({
              "county": "last",
              "state": "last",
              "reported_cases": "sum"
          })
          .rename(columns={"reported_cases": "new_cases_week"})
    )

    weekly = weekly.sort_values(["county_fips", "week"]).reset_index(drop=True)
    weekly["new_cases_week"] = weekly["new_cases_week"].clip(lower=0)
    weekly["cumulative_cases"] = weekly.groupby("county_fips")["new_cases_week"].cumsum()

    return weekly


# =========================
# STEP 2: COUNTY BASE
# =========================

def load_gazetteer(local_path: str = LOCAL_GAZETTEER_TXT) -> pd.DataFrame:
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Missing {local_path}")

    df = pd.read_csv(local_path, sep="|")
    df.columns = [c.strip() for c in df.columns]

    geoid_col = [c for c in df.columns if c.upper() == "GEOID"]
    if not geoid_col:
        raise ValueError("GEOID column missing in gazetteer file.")
    geoid_col = geoid_col[0]

    name_col = [c for c in df.columns if c.lower() == "name"]
    name_col = name_col[0] if name_col else None

    lat_col = [c for c in df.columns if "intptlat" in c.lower()]
    lon_col = [c for c in df.columns if "intptlong" in c.lower()]
    if not lat_col or not lon_col:
        raise ValueError("Latitude/longitude columns missing in gazetteer.")

    lat_col = lat_col[0]
    lon_col = lon_col[0]

    out = pd.DataFrame({
        "county_fips": df[geoid_col].astype(str).str.zfill(5),
        "county_name_gaz": df[name_col].astype(str) if name_col else "",
        "centroid_lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "centroid_lon": pd.to_numeric(df[lon_col], errors="coerce"),
    })
    out["state_fips"] = out["county_fips"].str[:2]
    out["state_abbr"] = out["state_fips"].map(STATE_ABBR)
    out["air_travel_proxy"] = out["state_abbr"].map(AIR_TRAVEL_PROXY).fillna(1.0)
    return out


def build_fallback_county_base_from_history(history: pd.DataFrame) -> pd.DataFrame:
    counties = history[["county_fips", "county", "state"]].drop_duplicates().copy()
    counties["state_abbr"] = counties["state"].apply(state_abbr_from_any)
    counties["county_name_gaz"] = counties["county"]
    counties["centroid_lat"] = np.nan
    counties["centroid_lon"] = np.nan
    counties["air_travel_proxy"] = counties["state_abbr"].map(AIR_TRAVEL_PROXY).fillna(1.0)
    return counties[[
        "county_fips", "county_name_gaz", "state_abbr",
        "centroid_lat", "centroid_lon", "air_travel_proxy"
    ]]


# =========================
# STEP 3: VACCINATION DATA
# =========================

def load_cdc_state_mmr(local_path: str = LOCAL_VACC_CSV) -> pd.DataFrame:
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Missing {local_path}")

    df = read_csv_flexible(local_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Vaccine/Exemption", "Dose", "Geography Type", "Geography", "School Year", "Estimate (%)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Vaccination file missing columns: {missing}")

    temp = df.copy()
    temp["Vaccine/Exemption"] = temp["Vaccine/Exemption"].astype(str).str.strip().str.lower()
    temp["Dose"] = temp["Dose"].astype(str).str.strip().str.lower()
    temp["Geography Type"] = temp["Geography Type"].astype(str).str.strip().str.lower()
    temp["Geography"] = temp["Geography"].astype(str).str.strip()
    temp["School Year"] = temp["School Year"].astype(str)
    temp["Estimate (%)"] = pd.to_numeric(temp["Estimate (%)"], errors="coerce")

    # MMR rows
    mmr = temp[
        temp["Vaccine/Exemption"].str.contains("mmr", na=False) &
        temp["Geography Type"].str.contains("state", na=False)
    ].copy()

    if mmr.empty:
        raise ValueError("No MMR state rows found in vaccination file.")

    # Prefer 2-dose if present
    mmr_2 = mmr[mmr["Dose"].str.contains("2", na=False)].copy()
    if not mmr_2.empty:
        mmr = mmr_2

    latest_year = mmr["School Year"].dropna().max()
    mmr = mmr[mmr["School Year"] == latest_year].copy()

    mmr_out = (
        mmr.groupby("Geography", as_index=False)["Estimate (%)"]
           .mean()
           .rename(columns={"Estimate (%)": "mmr_coverage_pct"})
    )
    mmr_out["state_abbr"] = mmr_out["Geography"].apply(state_abbr_from_any)

    # Exemption rows
    ex = temp[
        temp["Vaccine/Exemption"].str.contains("exemption", na=False) &
        temp["Geography Type"].str.contains("state", na=False)
    ].copy()

    if not ex.empty:
        latest_ex_year = ex["School Year"].dropna().max()
        ex = ex[ex["School Year"] == latest_ex_year].copy()

        # Prefer "Any Exemption" / total if present
        pref = ex[
            ex["Vaccine/Exemption"].str.contains("any", na=False) |
            ex["Vaccine/Exemption"].str.contains("total", na=False)
        ].copy()
        if not pref.empty:
            ex = pref

        ex_out = (
            ex.groupby("Geography", as_index=False)["Estimate (%)"]
              .mean()
              .rename(columns={"Estimate (%)": "exemption_pct"})
        )
        ex_out["state_abbr"] = ex_out["Geography"].apply(state_abbr_from_any)
    else:
        ex_out = pd.DataFrame(columns=["state_abbr", "exemption_pct"])

    out = mmr_out.merge(ex_out[["state_abbr", "exemption_pct"]], on="state_abbr", how="left")
    out = out[out["state_abbr"] != ""].drop_duplicates("state_abbr")
    out["mmr_coverage_pct"] = out["mmr_coverage_pct"].fillna(92.5)
    out["exemption_pct"] = out["exemption_pct"].fillna(3.6)
    out["susceptible_proxy"] = 100 - out["mmr_coverage_pct"]

    return out[["state_abbr", "mmr_coverage_pct", "exemption_pct", "susceptible_proxy"]]


# =========================
# STEP 4: COMMUTING FLOWS
# =========================

def load_commuting_flows(local_path: str = LOCAL_COMMUTER_CSV) -> pd.DataFrame:
    """
    Parse commuter.csv with a two-row header and duplicate labels.
    This version assigns columns by position after reconstructing the header.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Missing {local_path}")

    raw = read_csv_flexible(local_path, header=None, low_memory=False)

    if raw.shape[0] < 3:
        raise ValueError("commuter.csv looks too short.")

    top = raw.iloc[0].fillna("").astype(str).tolist()
    sub = raw.iloc[1].fillna("").astype(str).tolist()

    combined = []
    for a, b in zip(top, sub):
        a = a.strip()
        b = b.strip()
        if a and b:
            combined.append(f"{a}|{b}")
        elif b:
            combined.append(b)
        else:
            combined.append(a)

    df = raw.iloc[2:].copy().reset_index(drop=True)

    if len(combined) < 9:
        raise ValueError(f"Unexpected commuter.csv structure. Parsed columns: {combined}")

    df.columns = [f"col_{i}" for i in range(df.shape[1])]

    res_state_col = "col_0"
    res_county_col = "col_1"
    work_state_col = "col_4"
    work_county_col = "col_5"
    workers_col = "col_8"

    flow = pd.DataFrame({
        "res_state": pd.to_numeric(df[res_state_col], errors="coerce"),
        "res_county": pd.to_numeric(df[res_county_col], errors="coerce"),
        "work_state": pd.to_numeric(df[work_state_col], errors="coerce"),
        "work_county": pd.to_numeric(df[work_county_col], errors="coerce"),
        "workers": pd.to_numeric(
            df[workers_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        ),
    }).dropna()

    flow["origin_fips"] = (
        flow["res_state"].astype(int).astype(str).str.zfill(2) +
        flow["res_county"].astype(int).astype(str).str.zfill(3)
    )
    flow["dest_fips"] = (
        flow["work_state"].astype(int).astype(str).str.zfill(2) +
        flow["work_county"].astype(int).astype(str).str.zfill(3)
    )

    flow = flow.groupby(["origin_fips", "dest_fips"], as_index=False)["workers"].sum()
    flow = flow[flow["origin_fips"] != flow["dest_fips"]].copy()

    return flow

def build_fallback_commuting_flows(county_base: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["origin_fips", "dest_fips", "workers"])


def build_commuting_exposure(
    panel: pd.DataFrame,
    flows: pd.DataFrame,
    max_links_per_dest: int = 100
) -> pd.DataFrame:
    if flows is None or flows.empty:
        panel = panel.copy()
        panel["commuting_exposure"] = 0.0
        panel["inbound_workers"] = 0.0
        return panel

    valid_fips = set(panel["county_fips"].dropna().astype(str).unique().tolist())

    flows = flows.copy()
    flows = flows[flows["workers"] > 0].copy()
    flows = flows[flows["origin_fips"].isin(valid_fips) & flows["dest_fips"].isin(valid_fips)].copy()

    if flows.empty:
        panel = panel.copy()
        panel["commuting_exposure"] = 0.0
        panel["inbound_workers"] = 0.0
        return panel

    flows = (
        flows.sort_values(["dest_fips", "workers"], ascending=[True, False])
             .groupby("dest_fips")
             .head(max_links_per_dest)
             .copy()
    )
    flows["dest_total"] = flows.groupby("dest_fips")["workers"].transform("sum")
    flows["flow_weight"] = np.where(flows["dest_total"] > 0, flows["workers"] / flows["dest_total"], 0.0)

    lagged = panel[["county_fips", "week", "new_cases_week", "population_proxy"]].copy()
    lagged["incidence_per_100k"] = (
        100000 * lagged["new_cases_week"] / lagged["population_proxy"].replace(0, np.nan)
    )
    lagged["incidence_per_100k"] = lagged["incidence_per_100k"].fillna(0)
    lagged = lagged.rename(
        columns={
            "county_fips": "origin_fips",
            "incidence_per_100k": "origin_incidence_per_100k"
        }
    )

    merged = flows.merge(lagged, on="origin_fips", how="left")
    merged["weighted_exposure"] = merged["flow_weight"] * merged["origin_incidence_per_100k"]

    exposure = (
        merged.groupby(["dest_fips", "week"], as_index=False)
              .agg(
                  commuting_exposure=("weighted_exposure", "sum"),
                  inbound_workers=("workers", "sum")
              )
              .rename(columns={"dest_fips": "county_fips"})
    )

    out = panel.merge(exposure, on=["county_fips", "week"], how="left")
    out["commuting_exposure"] = out["commuting_exposure"].fillna(0.0)
    out["inbound_workers"] = out["inbound_workers"].fillna(0.0)
    return out


# =========================
# STEP 5: PANEL FEATURES
# =========================

def create_full_week_grid(weekly: pd.DataFrame, county_base: pd.DataFrame) -> pd.DataFrame:
    min_week = weekly["week"].min()
    max_week = weekly["week"].max()
    all_weeks = pd.date_range(min_week, max_week, freq="W-SAT")

    active_fips = set(weekly["county_fips"].dropna().astype(str).unique().tolist())

    counties = county_base[
        ["county_fips", "county_name_gaz", "state_abbr", "centroid_lat", "centroid_lon", "air_travel_proxy"]
    ].drop_duplicates().copy()
    counties = counties[counties["county_fips"].isin(active_fips)].copy()

    counties["key"] = 1
    weeks_df = pd.DataFrame({"week": all_weeks, "key": 1})
    grid = counties.merge(weeks_df, on="key").drop(columns="key")

    panel = grid.merge(
        weekly[["county_fips", "week", "county", "state", "cumulative_cases", "new_cases_week"]],
        on=["county_fips", "week"],
        how="left"
    )

    panel["county"] = panel["county"].fillna(panel["county_name_gaz"])
    panel["state"] = panel["state"].fillna(panel["state_abbr"])
    panel["cumulative_cases"] = panel.groupby("county_fips")["cumulative_cases"].ffill().fillna(0)
    panel["new_cases_week"] = panel["new_cases_week"].fillna(0)
    panel["population_proxy"] = 100000

    return panel


def add_local_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["county_fips", "week"]).copy()

    for lag in [1, 2, 3, 4]:
        panel[f"cases_lag_{lag}"] = panel.groupby("county_fips")["new_cases_week"].shift(lag)
        panel[f"cum_lag_{lag}"] = panel.groupby("county_fips")["cumulative_cases"].shift(lag)

    panel["cases_last_4w"] = sum(panel[f"cases_lag_{lag}"].fillna(0) for lag in [1, 2, 3, 4])
    panel["cases_trend"] = panel["cases_lag_1"].fillna(0) - panel["cases_lag_4"].fillna(0)
    panel["active_recent"] = (panel["cases_last_4w"] > 0).astype(int)
    panel["had_any_case_ever"] = (panel["cum_lag_1"].fillna(0) > 0).astype(int)
    panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype(int)
    panel["month"] = panel["week"].dt.month.astype(int)

    return panel


def add_spatial_neighbor_features(
    panel: pd.DataFrame,
    county_base: pd.DataFrame,
    k_neighbors: int = 8
) -> pd.DataFrame:
    cb = county_base[["county_fips", "centroid_lat", "centroid_lon"]].dropna().copy()

    if cb.empty:
        out = panel.copy()
        out["neighbor_exposure"] = 0.0
        out["avg_neighbor_distance"] = 100.0
        return out

    neighbors = []
    records = cb.to_dict(orient="records")

    for row in records:
        dists = []
        for row2 in records:
            if row["county_fips"] == row2["county_fips"]:
                continue
            d = haversine_miles(
                row["centroid_lat"], row["centroid_lon"],
                row2["centroid_lat"], row2["centroid_lon"]
            )
            dists.append((row2["county_fips"], d))

        dists.sort(key=lambda x: x[1])
        for nfips, dist in dists[:k_neighbors]:
            neighbors.append({
                "county_fips": row["county_fips"],
                "neighbor_fips": nfips,
                "distance_miles": dist
            })

    neigh = pd.DataFrame(neighbors)
    if neigh.empty:
        out = panel.copy()
        out["neighbor_exposure"] = 0.0
        out["avg_neighbor_distance"] = 100.0
        return out

    neigh["dist_weight"] = 1 / neigh["distance_miles"].replace(0, 1)

    lag = panel[["county_fips", "week", "new_cases_week", "population_proxy"]].copy()
    lag["neighbor_incidence"] = (
        100000 * lag["new_cases_week"] / lag["population_proxy"].replace(0, np.nan)
    )
    lag["neighbor_incidence"] = lag["neighbor_incidence"].fillna(0)
    lag = lag.rename(columns={"county_fips": "neighbor_fips"})

    merged = neigh.merge(lag, on="neighbor_fips", how="left")
    merged["weighted_neighbor_incidence"] = merged["dist_weight"] * merged["neighbor_incidence"]

    exposure = (
        merged.groupby(["county_fips", "week"], as_index=False)
        .agg(
            neighbor_exposure=("weighted_neighbor_incidence", "sum"),
            avg_neighbor_distance=("distance_miles", "mean")
        )
    )

    out = panel.merge(exposure, on=["county_fips", "week"], how="left")
    out["neighbor_exposure"] = out["neighbor_exposure"].fillna(0)
    out["avg_neighbor_distance"] = out["avg_neighbor_distance"].fillna(100.0)
    return out


def add_vaccination_features(panel: pd.DataFrame, mmr_state: pd.DataFrame) -> pd.DataFrame:
    out = panel.merge(mmr_state, on="state_abbr", how="left")
    out["mmr_coverage_pct"] = out["mmr_coverage_pct"].fillna(92.5)
    out["exemption_pct"] = out["exemption_pct"].fillna(3.6)
    out["susceptible_proxy"] = out["susceptible_proxy"].fillna(100 - out["mmr_coverage_pct"])
    return out


def finalize_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel["commuting_exposure"] = panel["commuting_exposure"].fillna(0)
    panel["inbound_workers"] = panel["inbound_workers"].fillna(0)
    panel["neighbor_exposure"] = panel["neighbor_exposure"].fillna(0)

    panel["travel_pressure"] = (
        panel["air_travel_proxy"].fillna(1.0) *
        (1 + np.log1p(panel["inbound_workers"]))
    )

    panel["importation_pressure"] = (
        0.65 * panel["commuting_exposure"] +
        0.35 * panel["neighbor_exposure"] * panel["air_travel_proxy"].fillna(1.0)
    )

    panel["spread_potential_index"] = (
        (panel["susceptible_proxy"] / 10.0) *
        (1 + panel["importation_pressure"]) *
        (1 + panel["cases_last_4w"])
    )

    return panel


# =========================
# STEP 6: TARGETS
# =========================

def make_targets(panel: pd.DataFrame, horizons: int) -> pd.DataFrame:
    panel = panel.sort_values(["county_fips", "week"]).copy()

    for h in range(1, horizons + 1):
        panel[f"target_new_cases_h{h}"] = panel.groupby("county_fips")["new_cases_week"].shift(-h)
        panel[f"target_cum_h{h}"] = panel.groupby("county_fips")["cumulative_cases"].shift(-h)

        prev = panel["cumulative_cases"]
        future = panel[f"target_cum_h{h}"]

        panel[f"target_growth_abs_h{h}"] = (future - prev).clip(lower=0)
        panel[f"target_growth_pct_h{h}"] = np.where(
            prev > 0, (future - prev) / prev, np.nan
        )

        panel[f"target_new_area_h{h}"] = (
            (panel["cumulative_cases"] <= 0) &
            (panel[f"target_cum_h{h}"] > 0)
        ).astype(int)

    return panel


# =========================
# STEP 7: MODELING
# =========================

FEATURE_COLS = [
    "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_4",
    "cum_lag_1", "cum_lag_2", "cum_lag_3", "cum_lag_4",
    "cases_last_4w", "cases_trend", "active_recent", "had_any_case_ever",
    "week_of_year", "month",
    "commuting_exposure", "inbound_workers", "neighbor_exposure",
    "avg_neighbor_distance", "mmr_coverage_pct", "exemption_pct",
    "susceptible_proxy", "air_travel_proxy", "travel_pressure",
    "importation_pressure", "spread_potential_index"
]


def train_forecast_models(panel: pd.DataFrame, horizons: int) -> Tuple[pd.DataFrame, Dict]:
    metrics = {}
    scored_frames = []

    usable = panel.copy()
    usable = usable.dropna(subset=["cases_lag_1"]).copy()

    # Reduce domination by all-zero rows while keeping some background zeros
    usable["recent_signal"] = (
        (usable["cases_last_4w"].fillna(0) > 0) |
        (usable["commuting_exposure"].fillna(0) > 0) |
        (usable["neighbor_exposure"].fillna(0) > 0)
    )

    signal_rows = usable[usable["recent_signal"]].copy()
    zero_rows = usable[~usable["recent_signal"]].copy()

    if len(signal_rows) > 0 and len(zero_rows) > 0:
        zero_sample = zero_rows.sample(
            n=min(len(zero_rows), max(len(signal_rows) * 3, 5000)),
            random_state=42
        )
        usable = pd.concat([signal_rows, zero_sample], ignore_index=True)

    usable = usable.sort_values(["week", "county_fips"]).reset_index(drop=True)

    if usable.empty:
        raise ValueError("No usable training rows. Need more time history.")

    split_week = usable["week"].quantile(0.8)
    train_df = usable[usable["week"] <= split_week].copy()
    test_df = usable[usable["week"] > split_week].copy()

    for h in range(1, horizons + 1):
        y_reg = f"target_growth_abs_h{h}"
        y_cls = f"target_new_area_h{h}"

        train_h = train_df.dropna(subset=[y_reg]).copy()
        test_h = test_df.dropna(subset=[y_reg]).copy()

        if train_h.empty:
            raise ValueError(f"No training rows available for horizon {h}.")

        X_train = train_h[FEATURE_COLS].fillna(0)
        X_test = test_h[FEATURE_COLS].fillna(0)

        # Log-transform sparse count target so model doesn't just collapse to zeros
        y_train_reg = np.log1p(train_h[y_reg].clip(lower=0))

        reg = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42
        )
        reg.fit(X_train, y_train_reg)

        if len(test_h) > 0:
            pred_abs = np.expm1(reg.predict(X_test))
            pred_abs = np.clip(pred_abs, a_min=0, a_max=None)
            mae = mean_absolute_error(test_h[y_reg], pred_abs)
        else:
            pred_abs = np.array([])
            mae = None

        train_cls = train_df.dropna(subset=[y_cls]).copy()
        test_cls = test_df.dropna(subset=[y_cls]).copy()

        Xc_train = train_cls[FEATURE_COLS].fillna(0)
        Xc_test = test_cls[FEATURE_COLS].fillna(0)

        clf = None
        if not train_cls.empty and train_cls[y_cls].nunique() > 1:
            pos = int(train_cls[y_cls].sum())
            neg = int(len(train_cls) - pos)
            scale_pos_weight = (neg / pos) if pos > 0 else 1.0

            clf = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                scale_pos_weight=scale_pos_weight
            )
            clf.fit(Xc_train, train_cls[y_cls])

            if len(test_cls) > 0:
                pred_new_area_prob = clf.predict_proba(Xc_test)[:, 1]
                auc = roc_auc_score(test_cls[y_cls], pred_new_area_prob) if test_cls[y_cls].nunique() > 1 else None
            else:
                auc = None
        else:
            auc = None

        metrics[f"h{h}"] = {
            "growth_abs_mae": None if mae is None else float(mae),
            "new_area_auc": None if auc is None else float(auc),
            "train_rows": int(len(train_h)),
            "test_rows": int(len(test_h)),
        }

        latest_week = usable["week"].max()
        latest = usable[usable["week"] == latest_week].copy()
        X_latest = latest[FEATURE_COLS].fillna(0)

        latest_pred_abs = np.expm1(reg.predict(X_latest))
        latest_pred_abs = np.clip(latest_pred_abs, a_min=0, a_max=None)

        latest_pred_pct = np.where(
            latest["cumulative_cases"] > 0,
            latest_pred_abs / latest["cumulative_cases"].replace(0, np.nan),
            np.nan
        )

        if clf is not None:
            latest_new_prob = clf.predict_proba(X_latest)[:, 1]
        else:
            latest_new_prob = np.zeros(len(X_latest))

        scored = latest[[
            "county_fips", "county", "state", "week", "cumulative_cases", "new_cases_week",
            "state_abbr", "centroid_lat", "centroid_lon"
        ]].copy()

        scored["horizon_weeks"] = h
        scored["forecast_week"] = scored["week"] + pd.to_timedelta(7 * h, unit="D")
        scored["expected_growth_abs"] = latest_pred_abs
        scored["expected_growth_pct"] = np.nan_to_num(
            latest_pred_pct, nan=0.0, posinf=0.0, neginf=0.0
        )
        scored["risk_new_cases_county"] = latest_new_prob

        # Better for sparse outbreak conditions:
        # emphasize emergence risk and transmission/importation context
        scored["risk_score"] = (
            0.35 * scored["risk_new_cases_county"] +
            0.20 * np.log1p(scored["expected_growth_abs"]) +
            0.10 * scored["expected_growth_pct"].clip(lower=0, upper=10) +
            0.15 * np.log1p(latest["importation_pressure"].fillna(0)) +
            0.10 * np.log1p(latest["cases_last_4w"].fillna(0)) +
            0.10 * (latest["susceptible_proxy"].fillna(7.5) / 10.0)
        )
        scored_frames.append(scored)

    forecast = pd.concat(scored_frames, ignore_index=True)
    return forecast, metrics

# =========================
# STEP 8: WEBSITE EXPORTS
# =========================

def load_county_geojson(local_path: str = LOCAL_COUNTY_GEOJSON) -> dict:
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Missing {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_web_outputs(forecast: pd.DataFrame, outdir: Path) -> None:
    web_data_dir = outdir / "web" / "data"
    safe_mkdir(web_data_dir)

    forecast = forecast.copy()
    forecast["week"] = forecast["week"].astype(str)
    forecast["forecast_week"] = forecast["forecast_week"].astype(str)

    forecast.to_csv(outdir / "county_measles_forecast.csv", index=False)

    forecast_clean = forecast.replace({np.nan: None})
    with open(web_data_dir / "forecast.json", "w", encoding="utf-8") as f:
        json.dump(forecast_clean.to_dict(orient="records"), f, indent=2, allow_nan=False)

    tables = {}
    for h in sorted(forecast["horizon_weeks"].unique()):
        dfh = forecast[forecast["horizon_weeks"] == h].copy()

        tables[str(h)] = {
            "top_risk_score": (
                dfh.sort_values("risk_score", ascending=False)
                   .head(50)[[
                       "county_fips", "county", "state",
                       "risk_score", "risk_new_cases_county",
                       "expected_growth_abs", "expected_growth_pct"
                   ]]
                   .replace({np.nan: None})
                   .to_dict(orient="records")
            ),
            "top_new_area_risk": (
                dfh.sort_values("risk_new_cases_county", ascending=False)
                   .head(50)[[
                       "county_fips", "county", "state",
                       "risk_score", "risk_new_cases_county",
                       "expected_growth_abs", "expected_growth_pct"
                   ]]
                   .replace({np.nan: None})
                   .to_dict(orient="records")
            ),
            "top_growth_abs": (
                dfh.sort_values("expected_growth_abs", ascending=False)
                   .head(50)[[
                       "county_fips", "county", "state",
                       "risk_score", "risk_new_cases_county",
                       "expected_growth_abs", "expected_growth_pct"
                   ]]
                   .replace({np.nan: None})
                   .to_dict(orient="records")
            )
        }

    with open(web_data_dir / "tables.json", "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, allow_nan=False)

    try:
        geojson = load_county_geojson()
    except Exception as e:
        print(f"Warning: could not load county GeoJSON for website map: {e}")
        geojson = {"type": "FeatureCollection", "features": []}

    h1 = forecast[forecast["horizon_weeks"] == forecast["horizon_weeks"].min()].copy()
    h1 = h1.replace({np.nan: None})
    lookup = {row["county_fips"]: row for _, row in h1.iterrows()}

    for feature in geojson.get("features", []):
        fips = str(feature["id"]).zfill(5)
        row = lookup.get(fips)
        props = feature.get("properties", {})
        props["county_fips"] = fips

        if row is not None:
            props["county"] = row["county"]
            props["state"] = row["state"]
            props["expected_growth_abs"] = 0.0 if row["expected_growth_abs"] is None else float(row["expected_growth_abs"])
            props["expected_growth_pct"] = 0.0 if row["expected_growth_pct"] is None else float(row["expected_growth_pct"])
            props["risk_new_cases_county"] = 0.0 if row["risk_new_cases_county"] is None else float(row["risk_new_cases_county"])
            props["risk_score"] = 0.0 if row["risk_score"] is None else float(row["risk_score"])
        else:
            props["county"] = props.get("NAME", "")
            props["state"] = ""
            props["expected_growth_abs"] = 0.0
            props["expected_growth_pct"] = 0.0
            props["risk_new_cases_county"] = 0.0
            props["risk_score"] = 0.0

        feature["properties"] = props

    with open(web_data_dir / "counties_h1.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson, f, allow_nan=False)

    horizon_dict = {}
    for h in sorted(forecast["horizon_weeks"].unique()):
        dfh = forecast[forecast["horizon_weeks"] == h].copy()
        dfh = dfh.replace({np.nan: None})
        horizon_dict[str(h)] = dfh.to_dict(orient="records")

    with open(web_data_dir / "forecast_by_horizon.json", "w", encoding="utf-8") as f:
        json.dump(horizon_dict, f, indent=2, allow_nan=False)

# =========================
# MAIN
# =========================

def run_pipeline(history_csv: Optional[str], outdir: str, horizons: int):
    out_path = Path(outdir)
    safe_mkdir(out_path)
    safe_mkdir(out_path / "web" / "data")

    print("Loading history...")
    hist_raw = load_history_csv(history_csv)
    hist = normalize_history(hist_raw)
    hist.to_csv(out_path / "measles_history_normalized.csv", index=False)

    print("Building weekly history...")
    weekly = build_weekly_history(hist)
    weekly.to_csv(out_path / "measles_weekly_county_history.csv", index=False)

    print("Loading county base...")
    try:
        county_base = load_gazetteer()
        county_base = county_base[county_base["county_fips"].isin(hist["county_fips"].unique())].copy()
        if county_base.empty:
            raise ValueError("Gazetteer loaded but no county FIPS overlapped with history.")
    except Exception as e:
        print(f"Warning: gazetteer load failed, using fallback county base. Error: {e}")
        county_base = build_fallback_county_base_from_history(hist)

    print("Loading CDC vaccination data...")
    try:
        mmr_state = load_cdc_state_mmr()
    except Exception as e:
        print(f"Warning: vaccination load failed, using defaults. Error: {e}")
        mmr_state = pd.DataFrame({
            "state_abbr": sorted(set(county_base["state_abbr"].dropna().astype(str))),
            "mmr_coverage_pct": 92.5,
            "exemption_pct": 3.6,
            "susceptible_proxy": 7.5,
        })

    print("Loading commuting flows...")
    try:
        flows = load_commuting_flows()
    except Exception as e:
        print(f"Warning: commuting load failed, continuing without commuting flows. Error: {e}")
        flows = build_fallback_commuting_flows(county_base)

    print("Creating county-week panel...")
    panel = create_full_week_grid(weekly, county_base)
    panel = add_local_lag_features(panel)
    panel = add_spatial_neighbor_features(panel, county_base)
    panel = add_vaccination_features(panel, mmr_state)
    panel = build_commuting_exposure(panel, flows)
    panel = finalize_features(panel)
    panel = make_targets(panel, horizons)

    panel.to_parquet(out_path / "county_week_model_panel.parquet", index=False)

    print("Training models...")
    forecast, metrics = train_forecast_models(panel, horizons)
    forecast.to_csv(out_path / "county_measles_forecast.csv", index=False)

    with open(out_path / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Writing website outputs...")
    write_web_outputs(forecast, out_path)

    print("Done.")
    print(f"Outputs written to: {out_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-csv", type=str, default=None, help="Local measles county CSV")
    parser.add_argument("--outdir", type=str, default="docs", help="Output directory")
    parser.add_argument("--horizons", type=int, default=4, help="Forecast horizons in weeks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        history_csv=args.history_csv,
        outdir=args.outdir,
        horizons=args.horizons
    )