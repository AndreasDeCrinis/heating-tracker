import csv
import calendar
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
import threading

import requests
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
READINGS_CSV = DATA_DIR / "readings.csv"
TARIFFS_CSV = DATA_DIR / "tariffs.csv"
OFFSETS_CSV = DATA_DIR / "offsets.csv"
GRID_POWER_CSV = DATA_DIR / "grid_power.csv"
WEATHER_CSV = DATA_DIR / "weather.csv"

CSV_DATE_FORMAT = "%d.%m.%Y"  # e.g. 01.01.2022

# Weather / HDD settings
WEATHER_CITY = "Arnfels"
WEATHER_LAT = 46.6764
WEATHER_LON = 15.4031
HDD_BASE_TEMP_C = 18.0
WEATHER_UPDATE_INTERVAL_SECONDS = 24 * 60 * 60  # once per day

# Only use data for HDD model from this date onwards
HDD_MODEL_START_DATE = date(2024, 9, 1)


# ---------- DATE HELPERS ----------

def parse_csv_date(s: str) -> date:
    """Parse a date from CSV (supports dd.mm.yyyy and yyyy-mm-dd)."""
    s = str(s).strip()
    # handle things like "01.01.2022;46286.0"
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    for fmt in (CSV_DATE_FORMAT, "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format in CSV: {s}")


def format_csv_date(d: date) -> str:
    """Format a date for CSV as dd.mm.yyyy."""
    return d.strftime(CSV_DATE_FORMAT)


# ---------- FILE SETUP ----------

def ensure_data_files():
    DATA_DIR.mkdir(exist_ok=True)

    if not READINGS_CSV.exists():
        with READINGS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "meter_reading"])

    if not TARIFFS_CSV.exists():
        with TARIFFS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "year",
                "base_price_per_kw",
                "contracted_power_kw",
                "working_price_per_kwh",
                "additional_yearly_costs",
            ])

    if not OFFSETS_CSV.exists():
        with OFFSETS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "offset_kwh"])

    if not GRID_POWER_CSV.exists():
        with GRID_POWER_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "power_kw"])

    # extended: avg, min, max
    if not WEATHER_CSV.exists():
        with WEATHER_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "avg_temp_c", "min_temp_c", "max_temp_c"])


# ---------- GENERIC SIMPLE CSV LOADERS (HANDLE ; OR ,) ----------

def _load_two_column_date_float_csv(path, col1_name, col2_name):
    """
    Robust reader for 'date;value' or 'date,value' CSVs, possibly mixed.
    Returns list of dicts with keys col1_name (date) and col2_name (float).
    Skips empty/broken lines.
    """
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Skip header if it contains the column name "date"
            if line_no == 0 and "date" in line.lower():
                continue

            sep = ";" if ";" in line else ","
            parts = line.split(sep)
            if len(parts) < 2:
                continue

            date_str = parts[0].strip()
            value_str = parts[1].strip()

            try:
                dt = parse_csv_date(date_str)
                val = float(value_str.replace(",", "."))
            except Exception:
                continue

            rows.append({col1_name: dt, col2_name: val})

    rows.sort(key=lambda r: r[col1_name])
    return rows


# ---------- READINGS ----------

def load_readings():
    return _load_two_column_date_float_csv(READINGS_CSV, "date", "meter_reading")


def append_reading(date_str, meter_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    meter = float(meter_str.replace(",", "."))
    with READINGS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([format_csv_date(dt), meter])


def save_readings(readings_list):
    """Overwrite readings.csv from a list of {'date', 'meter_reading'} dicts."""
    readings_sorted = sorted(readings_list, key=lambda r: r["date"])
    with READINGS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "meter_reading"])
        for r in readings_sorted:
            writer.writerow([format_csv_date(r["date"]), r["meter_reading"]])


# ---------- TARIFFS ----------

def load_tariffs():
    if not TARIFFS_CSV.exists():
        return {}
    df = pd.read_csv(TARIFFS_CSV)
    tariffs = {}
    for _, row in df.iterrows():
        year = int(row["year"])
        tariffs[year] = {
            "year": year,
            "base_price_per_kw": float(row["base_price_per_kw"]),
            "contracted_power_kw": float(row["contracted_power_kw"]),
            "working_price_per_kwh": float(row["working_price_per_kwh"]),
            "additional_yearly_costs": float(row["additional_yearly_costs"]),
        }
    return tariffs


def save_tariffs(tariff_list):
    df = pd.DataFrame(tariff_list)
    df.to_csv(TARIFFS_CSV, index=False)


# ---------- OFFSETS (meter replacement) ----------

def load_offsets():
    return _load_two_column_date_float_csv(OFFSETS_CSV, "date", "offset_kwh")


def append_offset(date_str, offset_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    offset = float(offset_str.replace(",", "."))
    with OFFSETS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([format_csv_date(dt), offset])


# ---------- GRID CONNECTION POWER (kW over time) ----------

def load_grid_power():
    return _load_two_column_date_float_csv(GRID_POWER_CSV, "date", "power_kw")


def append_grid_power(date_str, power_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    power_kw = float(power_str.replace(",", "."))
    with GRID_POWER_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([format_csv_date(dt), power_kw])


# ---------- WEATHER HISTORY (avg + min + max) ----------

def load_weather_history():
    """
    Load weather.csv as dict[date -> {"avg": float, "min": float|None, "max": float|None}].

    Supports old files that only had avg_temp_c.
    """
    if not WEATHER_CSV.exists():
        return {}

    history = {}
    with WEATHER_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        has_min = "min_temp_c" in fieldnames
        has_max = "max_temp_c" in fieldnames

        for row in reader:
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
                avg = float(row["avg_temp_c"])
            except Exception:
                continue

            min_t = None
            max_t = None
            if has_min:
                val = row.get("min_temp_c")
                if val not in (None, ""):
                    try:
                        min_t = float(val)
                    except Exception:
                        pass
            if has_max:
                val = row.get("max_temp_c")
                if val not in (None, ""):
                    try:
                        max_t = float(val)
                    except Exception:
                        pass

            history[d] = {"avg": avg, "min": min_t, "max": max_t}
    return history


def save_weather_history(history: dict):
    """
    Save weather history with 3 columns: avg, min, max.
    """
    with WEATHER_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "avg_temp_c", "min_temp_c", "max_temp_c"])
        for d in sorted(history.keys()):
            info = history[d]
            avg = info.get("avg")
            min_t = info.get("min")
            max_t = info.get("max")
            writer.writerow([
                d.isoformat(),
                "" if avg is None else avg,
                "" if min_t is None else min_t,
                "" if max_t is None else max_t,
            ])


def fetch_historical_weather(start_date: date, end_date: date) -> dict:
    """
    Fetch historical daily mean, min, max temperature from Open-Meteo
    for Arnfels between start_date and end_date (inclusive).

    Returns dict[date -> {"avg": float, "min": float|None, "max": float|None}]
    """
    logger.info(f"Fetching historical weather {start_date}..{end_date} for {WEATHER_CITY}")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max",
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        times = daily.get("time", [])
        temps_mean = daily.get("temperature_2m_mean", [])
        temps_min = daily.get("temperature_2m_min", [])
        temps_max = daily.get("temperature_2m_max", [])
        result = {}

        for i, t_str in enumerate(times):
            try:
                d = datetime.strptime(t_str, "%Y-%m-%d").date()
            except Exception:
                continue
            try:
                avg = float(temps_mean[i])
            except Exception:
                avg = None
            try:
                mn = float(temps_min[i])
            except Exception:
                mn = None
            try:
                mx = float(temps_max[i])
            except Exception:
                mx = None

            if avg is not None:
                result[d] = {"avg": avg, "min": mn, "max": mx}

        logger.info(f"Fetched {len(result)} historical days")
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch historical weather: {e}")
        return {}


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def update_weather_history_if_needed():
    """
    Ensure weather.csv has daily temperature data for all dates
    between the earliest reading date and today.
    """
    readings = load_readings()
    if not readings:
        return

    history = load_weather_history()
    have_dates = set(history.keys())

    earliest = min(r["date"] for r in readings)
    today = date.today()
    if earliest > today:
        return

    missing_ranges = []
    current_start = None
    current_end = None

    for d in daterange(earliest, today):
        if d not in have_dates:
            if current_start is None:
                current_start = d
                current_end = d
            else:
                current_end = d
        else:
            if current_start is not None:
                missing_ranges.append((current_start, current_end))
                current_start = None
                current_end = None
    if current_start is not None:
        missing_ranges.append((current_start, current_end))

    if not missing_ranges:
        logger.info("Weather history up to date.")
        return

    logger.info(f"Weather history missing {len(missing_ranges)} ranges")
    for start_d, end_d in missing_ranges:
        fetched = fetch_historical_weather(start_d, end_d)
        history.update(fetched)

    save_weather_history(history)
    logger.info("Weather history updated.")


def schedule_daily_weather_update():
    """Schedule daily weather history update in a background thread."""

    def _update_and_reschedule():
        try:
            update_weather_history_if_needed()
        except Exception as e:
            logger.error(f"Error updating weather history: {e}")
        finally:
            threading.Timer(WEATHER_UPDATE_INTERVAL_SECONDS, _update_and_reschedule).start()

    threading.Timer(60, _update_and_reschedule).start()


# ---------- WEATHER FORECAST (mean only; used for HDD prediction) ----------

def fetch_weather_forecast(days: int):
    """
    Fetch daily mean temperature forecast for Arnfels using Open-Meteo.
    Returns list[(date, avg_temp_c)] for up to `days`.
    """
    if days <= 0:
        return []

    base_days = min(days, 16)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "daily": "temperature_2m_mean",
        "forecast_days": base_days,
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        times = daily.get("time", [])
        temps = daily.get("temperature_2m_mean", [])
        result = []
        for t_str, temp in zip(times, temps):
            try:
                d = datetime.strptime(t_str, "%Y-%m-%d").date()
                result.append((d, float(temp)))
            except Exception:
                continue

        if days > base_days:
            if result:
                avg_temp = sum(t for _, t in result) / len(result)
                last_date = result[-1][0]
            else:
                avg_temp = 5.0
                last_date = date.today()
            for _ in range(days - base_days):
                last_date = last_date + timedelta(days=1)
                result.append((last_date, avg_temp))

        return result[:days]
    except Exception as e:
        logger.warning(f"Failed to fetch forecast: {e}")
        return []


def build_kwh_forecast_from_temps(temp_list, model, tariffs_by_year, grid_powers_sorted):
    """
    Use HDD model to turn a list of (date, avg_temp_c) into
    daily kWh and cost predictions.

    Returns:
      total_kwh, total_cost, avg_kwh_per_day, avg_cost_per_day, per_day_list
    """
    if not model or not temp_list:
        return None, None, None, None, []

    base_load = model["base_load"]
    alpha = model["alpha"]

    total_kwh = 0.0
    total_cost = 0.0
    per_day = []

    for d, temp in temp_list:
        hdd = max(0.0, HDD_BASE_TEMP_C - temp)
        kwh = max(0.0, base_load + alpha * hdd)

        year = d.year
        tariff = tariffs_by_year.get(year)
        cost = None
        if tariff:
            days_in_year = 366 if calendar.isleap(year) else 365
            power_kw = get_power_kw_for_date(d, grid_powers_sorted, tariffs_by_year)
            yearly_fixed = (tariff["base_price_per_kw"] * power_kw
                            + tariff["additional_yearly_costs"])
            fixed_per_day = yearly_fixed / days_in_year
            energy_cost_per_day = kwh * tariff["working_price_per_kwh"]
            cost = fixed_per_day + energy_cost_per_day

        per_day.append({"date": d, "kwh": kwh, "cost": cost})
        total_kwh += kwh
        if cost is not None:
            total_cost += cost

    days = len(temp_list)
    avg_kwh = total_kwh / days if days > 0 else None
    avg_cost = total_cost / days if (days > 0 and total_cost > 0) else None

    return total_kwh, total_cost, avg_kwh, avg_cost, per_day


# ---------- VIRTUAL READINGS (apply offsets) ----------

def compute_virtual_readings(readings, offsets):
    """Apply offsets to handle meter replacements / resets."""
    if not readings:
        return []
    offsets_sorted = sorted(offsets, key=lambda o: o["date"])
    virtual_readings = []
    for r in sorted(readings, key=lambda r: r["date"]):
        total_offset = 0.0
        for o in offsets_sorted:
            if o["date"] <= r["date"]:
                total_offset += o["offset_kwh"]
        virtual_readings.append({
            "date": r["date"],
            "meter_reading": r["meter_reading"],
            "virtual_meter": r["meter_reading"] + total_offset,
        })
    return virtual_readings


def get_power_kw_for_date(dt, grid_powers_sorted, tariffs_by_year):
    """
    Return the connected power (kW) valid for a given date.
    - Use the latest grid_power entry with date <= dt
    - If none, fall back to the tariff's contracted_power_kw for that year (if set)
    """
    power_kw = None
    for gp in grid_powers_sorted:
        if gp["date"] <= dt:
            power_kw = gp["power_kw"]
        else:
            break

    if power_kw is None:
        tariff = tariffs_by_year.get(dt.year)
        if tariff:
            power_kw = float(tariff.get("contracted_power_kw", 0.0))
        else:
            power_kw = 0.0

    return power_kw


# ---------- STATS & COSTS + WEATHER/HDD ----------

def compute_stats(readings, tariffs_by_year, offsets, grid_powers, weather_history):
    """
    Main stats computation.
    """
    virtual_readings = compute_virtual_readings(readings, offsets)
    today = date.today()
    current_year = today.year

    empty_result = {
        "daily_entries": [],
        "all_time_avg_kwh_per_day": None,
        "all_time_avg_cost_per_day": None,
        "monthly_stats": [],
        "yearly_stats": [],
        "current_year": current_year,
        "current_year_months": [],
        "month_comparisons": [],
        "last_month_avg_kwh_per_day": None,
        "current_month_avg_kwh_per_day": None,
        "current_heating_season_avg_kwh_per_day": None,
        "last_heating_season_avg_kwh_per_day": None,
        "heating_season_labels": ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                                  "Dec", "Jan", "Feb", "Mar", "Apr", "May"],
        "heating_season_series": [],
        "heating_season_forecast_series": [None] * 12,
        "forecast_7d_total_kwh": None,
        "forecast_7d_avg_kwh_per_day": None,
        "forecast_current_month_total_kwh": None,
        "forecast_current_month_avg_kwh_per_day": None,
        "heating_model": None,
        "temp_heating_season_series": [],
        "model_season_comparisons": [],
        "model_season_monthly_series": [],
    }

    if len(virtual_readings) < 2:
        return empty_result

    daily_entries = []
    grid_powers_sorted = sorted(grid_powers, key=lambda g: g["date"])

    # --- per-interval entries (for charts) ---
    for prev, curr in zip(virtual_readings, virtual_readings[1:]):
        days = (curr["date"] - prev["date"]).days
        if days <= 0:
            continue

        delta_kwh = curr["virtual_meter"] - prev["virtual_meter"]
        if delta_kwh < 0:
            # meter rolled over or reset without offset
            delta_kwh = curr["meter_reading"]

        daily_kwh = delta_kwh / days

        start_date = prev["date"]
        end_date = curr["date"]

        year = start_date.year
        tariff = tariffs_by_year.get(year)
        daily_cost = None

        if tariff:
            days_in_year = 366 if calendar.isleap(year) else 365
            power_kw = get_power_kw_for_date(start_date, grid_powers_sorted, tariffs_by_year)
            yearly_fixed = (tariff["base_price_per_kw"] * power_kw
                            + tariff["additional_yearly_costs"])
            fixed_per_day = yearly_fixed / days_in_year
            energy_cost_per_day = daily_kwh * tariff["working_price_per_kwh"]
            daily_cost = fixed_per_day + energy_cost_per_day

        total_cost_interval = daily_cost * days if daily_cost is not None else None

        daily_entries.append({
            "date": end_date,
            "start_date": start_date,
            "days": days,
            "delta_kwh": delta_kwh,
            "daily_kwh": daily_kwh,
            "daily_cost": daily_cost,
            "total_cost": total_cost_interval,
        })

    if not daily_entries:
        return empty_result

    # --- per-day list for HDD model & season comparison ---
    per_day_usage = []
    for e in daily_entries:
        kwh_day = e["daily_kwh"]
        d = e["start_date"]
        while d < e["date"]:
            per_day_usage.append({"date": d, "kwh": kwh_day})
            d += timedelta(days=1)

    # --- fit HDD model: kWh/day = base_load + alpha * HDD ---
    hdd_x = []
    kwh_y = []

    for pd in per_day_usage:
        d = pd["date"]
        # Only use data from HDD_MODEL_START_DATE onwards for training
        if d < HDD_MODEL_START_DATE:
            continue
        kwh = pd["kwh"]
        info = weather_history.get(d)
        if not info:
            continue
        temp = info.get("avg")
        if temp is None:
            continue
        hdd = max(0.0, HDD_BASE_TEMP_C - temp)
        if hdd <= 0:
            continue
        hdd_x.append(hdd)
        kwh_y.append(kwh)

    heating_model = None
    if len(hdd_x) >= 10 and len(set(hdd_x)) > 1:
        n = len(hdd_x)
        x_mean = sum(hdd_x) / n
        y_mean = sum(kwh_y) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(hdd_x, kwh_y))
        den = sum((x - x_mean) ** 2 for x in hdd_x)
        if den > 0:
            alpha = num / den
            base_load = y_mean - alpha * x_mean
            heating_model = {
                "base_load": base_load,
                "alpha": alpha,
                "n_days": n,
                "hdd_base_temp": HDD_BASE_TEMP_C,
                "model_start": HDD_MODEL_START_DATE,
            }
            logger.info(
                f"HDD model fitted (from {HDD_MODEL_START_DATE}): "
                f"base_load={base_load:.2f} kWh/d, alpha={alpha:.2f} kWh/deg-day, days={n}"
            )
        else:
            logger.info("HDD model: variance of HDD is zero, cannot fit.")
    else:
        logger.info("HDD model: not enough recent data to fit (need >=10 heating days).")

    # --- split intervals across months/years ---

    yearly_groups = {}
    monthly_groups = {}
    total_days = 0
    total_kwh_all = 0.0
    total_cost_all = 0.0

    for e in daily_entries:
        daily_kwh = e["daily_kwh"]
        daily_cost = e["daily_cost"]
        cursor = e["start_date"]
        end = e["date"]

        while cursor < end:
            year = cursor.year
            month = cursor.month

            if month == 12:
                next_month_first = date(year + 1, 1, 1)
            else:
                next_month_first = date(year, month + 1, 1)

            chunk_end = min(end, next_month_first)
            days_chunk = (chunk_end - cursor).days
            if days_chunk <= 0:
                break

            kwh_chunk = daily_kwh * days_chunk
            cost_chunk = daily_cost * days_chunk if daily_cost is not None else None

            yg = yearly_groups.setdefault(year, {"total_kwh": 0.0, "total_cost": 0.0, "days": 0})
            yg["total_kwh"] += kwh_chunk
            yg["days"] += days_chunk
            if cost_chunk is not None:
                yg["total_cost"] += cost_chunk

            month_key = f"{year}-{month:02d}"
            mg = monthly_groups.setdefault(month_key, {"total_kwh": 0.0, "total_cost": 0.0, "days": 0})
            mg["total_kwh"] += kwh_chunk
            mg["days"] += days_chunk
            if cost_chunk is not None:
                mg["total_cost"] += cost_chunk

            total_days += days_chunk
            total_kwh_all += kwh_chunk
            if cost_chunk is not None:
                total_cost_all += cost_chunk

            cursor = chunk_end

    all_time_avg_kwh = total_kwh_all / total_days if total_days > 0 else None
    all_time_avg_cost = (total_cost_all / total_days) if (total_days > 0 and total_cost_all > 0) else None

    # --- monthly_stats ---
    monthly_stats = []
    for month_key in sorted(monthly_groups.keys()):
        group = monthly_groups[month_key]
        days_m = group["days"]
        total_kwh_m = group["total_kwh"]
        total_cost_m = group["total_cost"] if group["total_cost"] != 0 else None
        avg_kwh_m = (total_kwh_m / days_m) if days_m > 0 else None
        avg_cost_m = (total_cost_m / days_m) if (days_m > 0 and total_cost_m is not None) else None
        monthly_stats.append({
            "month": month_key,
            "days": days_m,
            "total_kwh": total_kwh_m,
            "total_cost": total_cost_m,
            "avg_kwh_per_day": avg_kwh_m,
            "avg_cost_per_day": avg_cost_m,
        })

    # --- yearly_stats ---
    yearly_stats = []
    for y in sorted(yearly_groups.keys()):
        g = yearly_groups[y]
        days_y = g["days"]
        total_kwh_y = g["total_kwh"]
        total_cost_y = g["total_cost"] if g["total_cost"] != 0 else None
        avg_kwh_y = (total_kwh_y / days_y) if days_y > 0 else None
        avg_cost_y = (total_cost_y / days_y) if (total_cost_y is not None and days_y > 0) else None
        yearly_stats.append({
            "year": y,
            "days": g["days"],
            "total_kwh": total_kwh_y,
            "total_cost": total_cost_y,
            "avg_kwh_per_day": avg_kwh_y,
            "avg_cost_per_day": avg_cost_y,
        })

    # --- current year monthly detail ---
    current_year_months = [m for m in monthly_stats if m["month"].startswith(f"{current_year}-")]

    # --- month comparisons (same calendar month across years) ---
    month_comparisons = []
    for mnum in range(1, 13):
        month_str = f"{mnum:02d}"
        stats_for_month = [m for m in monthly_stats if m["month"][5:7] == month_str]
        if not stats_for_month:
            continue
        stats_for_month.sort(key=lambda m: m["month"])
        month_comparisons.append({
            "month_num": month_str,
            "month_name": calendar.month_name[mnum],
            "entries": stats_for_month,
        })

    # --- extra summary metrics: current & last month ---
    current_month_key = today.strftime("%Y-%m")
    current_year_int, current_month_int = map(int, current_month_key.split("-"))

    current_month_avg = None
    for m in monthly_stats:
        if m["month"] == current_month_key:
            current_month_avg = m["avg_kwh_per_day"]
            break

    last_month_avg = None
    last_before = None
    for m in monthly_stats:
        y_m, mo_m = map(int, m["month"].split("-"))
        if (y_m, mo_m) < (current_year_int, current_month_int):
            last_before = m
    if last_before is not None:
        last_month_avg = last_before["avg_kwh_per_day"]

    # --- heating seasons (1.6.–31.5.) ---
    season_groups = {}
    season_series_map = {}

    for m in monthly_stats:
        year_m, month_m = map(int, m["month"].split("-"))
        if month_m >= 6:
            season_start = year_m
        else:
            season_start = year_m - 1

        sg = season_groups.setdefault(season_start, {"days": 0, "total_kwh": 0.0, "total_cost": 0.0})
        sg["days"] += m["days"]
        sg["total_kwh"] += m["total_kwh"]
        if m["total_cost"] is not None:
            sg["total_cost"] += m["total_cost"]

        series = season_series_map.setdefault(season_start, {
            "label": f"{season_start}/{(season_start + 1) % 100:02d}",
            "start_year": season_start,
            "data": [None] * 12,
        })

        if month_m >= 6:
            idx = month_m - 6
        else:
            idx = month_m + 6

        series["data"][idx] = m["avg_kwh_per_day"]

    if today.month >= 6:
        current_season_start = today.year
    else:
        current_season_start = today.year - 1
    last_season_start = current_season_start - 1

    def season_avg(start_year):
        sg = season_groups.get(start_year)
        if not sg or sg["days"] <= 0:
            return None
        return sg["total_kwh"] / sg["days"]

    current_season_avg = season_avg(current_season_start)
    last_season_avg = season_avg(last_season_start)

    heating_season_labels = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                             "Dec", "Jan", "Feb", "Mar", "Apr", "May"]
    heating_season_series = [
        series for sy, series in sorted(season_series_map.items())
        if sy >= 2024
    ]

    # ---------- Forecasts ----------

    forecast_7d_total_kwh = None
    forecast_7d_avg_kwh = None
    forecast_current_month_total_kwh = None
    forecast_current_month_avg_kwh = None
    heating_season_forecast_series = [None] * 12

    if heating_model:
        # --- 7-day rolling forecast ---
        temps_7 = fetch_weather_forecast(7)
        t7_kwh, t7_cost, t7_avg_kwh, t7_avg_cost, per_day_7 = build_kwh_forecast_from_temps(
            temps_7, heating_model, tariffs_by_year, grid_powers_sorted
        )
        forecast_7d_total_kwh = t7_kwh
        forecast_7d_avg_kwh = t7_avg_kwh

        # --- current month forecast (calendar month) ---
        first_of_month = date(today.year, today.month, 1)
        if today.month == 12:
            next_month_first = date(today.year + 1, 1, 1)
        else:
            next_month_first = date(today.year, today.month + 1, 1)
        last_of_month = next_month_first - timedelta(days=1)

        days_for_forecast = (last_of_month - today).days + 1 if last_of_month >= today else 0
        forecast_temps = fetch_weather_forecast(days_for_forecast) if days_for_forecast > 0 else []
        forecast_temp_map = {d: t for d, t in forecast_temps}

        temps_month = []
        cur = first_of_month
        while cur <= last_of_month:
            if cur <= today:
                info = weather_history.get(cur)
                if info and info.get("avg") is not None:
                    temps_month.append((cur, info["avg"]))
            else:
                t = forecast_temp_map.get(cur)
                if t is not None:
                    temps_month.append((cur, t))
            cur += timedelta(days=1)

        tmonth_kwh, tmonth_cost, tmonth_avg_kwh, tmonth_avg_cost, per_day_month = \
            build_kwh_forecast_from_temps(temps_month, heating_model, tariffs_by_year, grid_powers_sorted)

        forecast_current_month_total_kwh = tmonth_kwh
        forecast_current_month_avg_kwh = tmonth_avg_kwh

        # Map this month's forecasted kWh/day into heating-season chart (one point)
        if per_day_month:
            if today.month >= 6:
                season_start_this_month = today.year
                idx = today.month - 6
            else:
                season_start_this_month = today.year - 1
                idx = today.month + 6

            if season_start_this_month == current_season_start and 0 <= idx < 12:
                month_days = len(per_day_month)
                if month_days > 0:
                    avg_month_kwh = tmonth_kwh / month_days
                    heating_season_forecast_series[idx] = avg_month_kwh

    # ---------- Temperature heating-season series (avg only) ----------

    temp_season_agg = {}
    for d, info in weather_history.items():
        avg = info.get("avg")
        if avg is None:
            continue

        if d.month >= 6:
            season_start = d.year
            idx = d.month - 6
        else:
            season_start = d.year - 1
            idx = d.month + 6

        if not (0 <= idx < 12):
            continue

        agg = temp_season_agg.setdefault(season_start, {
            "sum": [0.0] * 12,
            "count": [0] * 12,
        })
        agg["sum"][idx] += avg
        agg["count"][idx] += 1

    temp_heating_season_series = []
    for sy in sorted(temp_season_agg.keys()):
        agg = temp_season_agg[sy]
        sums = agg["sum"]
        counts = agg["count"]
        data = []
        for i in range(12):
            if counts[i] > 0:
                data.append(sums[i] / counts[i])
            else:
                data.append(None)
        temp_heating_season_series.append({
            "label": f"{sy}/{(sy + 1) % 100:02d}",
            "start_year": sy,
            "data": data,
        })

    temp_heating_season_series = [
        s for s in temp_heating_season_series if s["start_year"] >= 2024
    ]

    # ---------- Model vs actual comparison per heating season ----------

    model_season_comparisons = []
    model_season_monthly_series = []

    if heating_model:
        base_load = heating_model["base_load"]
        alpha = heating_model["alpha"]
        comparison_agg = {}
        monthly_agg = {}

        for pd in per_day_usage:
            d = pd["date"]
            kwh_actual = pd["kwh"]
            info = weather_history.get(d)
            if not info:
                continue
            temp = info.get("avg")
            if temp is None:
                continue

            hdd = max(0.0, HDD_BASE_TEMP_C - temp)
            kwh_pred = max(0.0, base_load + alpha * hdd)

            if d.month >= 6:
                season_start = d.year
                idx = d.month - 6
            else:
                season_start = d.year - 1
                idx = d.month + 6

            if not (0 <= idx < 12):
                continue

            agg = comparison_agg.setdefault(season_start, {
                "actual_total_kwh": 0.0,
                "predicted_total_kwh": 0.0,
                "days": 0,
            })
            agg["actual_total_kwh"] += kwh_actual
            agg["predicted_total_kwh"] += kwh_pred
            agg["days"] += 1

            mdata = monthly_agg.setdefault(season_start, {
                "actual_sum": [0.0] * 12,
                "pred_sum": [0.0] * 12,
                "days": [0] * 12,
            })
            mdata["actual_sum"][idx] += kwh_actual
            mdata["pred_sum"][idx] += kwh_pred
            mdata["days"][idx] += 1

        for sy in sorted(comparison_agg.keys()):
            agg = comparison_agg[sy]
            if agg["days"] < 10:
                continue

            actual = agg["actual_total_kwh"]
            predicted = agg["predicted_total_kwh"]
            diff = predicted - actual
            pct = (diff / actual * 100.0) if actual > 0 else None

            model_season_comparisons.append({
                "season_label": f"{sy}/{(sy + 1) % 100:02d}",
                "season_start": sy,
                "days": agg["days"],
                "actual_total_kwh": actual,
                "predicted_total_kwh": predicted,
                "diff_kwh": diff,
                "diff_pct": pct,
            })

            mdata = monthly_agg.get(sy)
            if mdata:
                monthly_actual = []
                monthly_pred = []
                for i in range(12):
                    if mdata["days"][i] > 0:
                        monthly_actual.append(mdata["actual_sum"][i] / mdata["days"][i])
                        monthly_pred.append(mdata["pred_sum"][i] / mdata["days"][i])
                    else:
                        monthly_actual.append(None)
                        monthly_pred.append(None)
            else:
                monthly_actual = [None] * 12
                monthly_pred = [None] * 12

            model_season_monthly_series.append({
                "season_label": f"{sy}/{(sy + 1) % 100:02d}",
                "season_start": sy,
                "actual": monthly_actual,
                "predicted": monthly_pred,
            })

    stats = {
        "daily_entries": daily_entries,
        "all_time_avg_kwh_per_day": all_time_avg_kwh,
        "all_time_avg_cost_per_day": all_time_avg_cost,
        "monthly_stats": monthly_stats,
        "yearly_stats": yearly_stats,
        "current_year": current_year,
        "current_year_months": current_year_months,
        "month_comparisons": month_comparisons,
        "last_month_avg_kwh_per_day": last_month_avg,
        "current_month_avg_kwh_per_day": current_month_avg,
        "current_heating_season_avg_kwh_per_day": current_season_avg,
        "last_heating_season_avg_kwh_per_day": last_season_avg,
        "heating_season_labels": heating_season_labels,
        "heating_season_series": heating_season_series,
        "heating_season_forecast_series": heating_season_forecast_series,
        "forecast_7d_total_kwh": forecast_7d_total_kwh,
        "forecast_7d_avg_kwh_per_day": forecast_7d_avg_kwh,
        "forecast_current_month_total_kwh": forecast_current_month_total_kwh,
        "forecast_current_month_avg_kwh_per_day": forecast_current_month_avg_kwh,
        "heating_model": heating_model,
        "temp_heating_season_series": temp_heating_season_series,
        "model_season_comparisons": model_season_comparisons,
        "model_season_monthly_series": model_season_monthly_series,
    }
    return stats


# ---------- ROUTES ----------

@app.route("/")
def index():
    """Dashboard / visualization."""
    ensure_data_files()

    try:
        update_weather_history_if_needed()
    except Exception as e:
        logger.error(f"Weather history update in index failed: {e}")

    readings = load_readings()
    tariffs = load_tariffs()
    offsets = load_offsets()
    grid_powers = load_grid_power()
    weather_history = load_weather_history()

    stats = compute_stats(readings, tariffs, offsets, grid_powers, weather_history)

    daily_entries = stats["daily_entries"][-60:]
    daily_labels = [e["date"].isoformat() for e in daily_entries]
    daily_values = [round(e["daily_kwh"], 2) for e in daily_entries]

    monthly_labels = [m["month"] for m in stats["monthly_stats"]]
    monthly_values = [
        round(m["avg_kwh_per_day"], 2) if m["avg_kwh_per_day"] is not None else None
        for m in stats["monthly_stats"]
    ]

    return render_template(
        "index.html",
        readings=readings[-20:],
        stats=stats,
        daily_labels=daily_labels,
        daily_values=daily_values,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        weather_city=WEATHER_CITY,
    )


@app.route("/input", methods=["GET", "POST"])
def input_view():
    """Data entry: readings + offsets."""
    ensure_data_files()

    if request.method == "POST":
        action = request.form.get("action", "reading")
        if action == "reading":
            date_str = request.form.get("date") or date.today().isoformat()
            meter_str = request.form.get("meter_reading", "").replace(",", ".")
            try:
                append_reading(date_str, meter_str)
                flash("Reading saved.", "success")
            except Exception as e:
                flash(f"Could not save reading: {e}", "danger")
        elif action == "offset":
            offset_date = request.form.get("offset_date")
            offset_value = request.form.get("offset_kwh", "").replace(",", ".")
            try:
                append_offset(offset_date, offset_value)
                flash("Offset saved.", "success")
            except Exception as e:
                flash(f"Could not save offset: {e}", "danger")

        return redirect(url_for("input_view"))

    readings = load_readings()
    offsets = load_offsets()
    grid_powers = load_grid_power()

    return render_template(
        "input.html",
        readings=readings[-10:],
        offsets=offsets,
        grid_powers=grid_powers,
        today=date.today().isoformat(),
    )


@app.route("/tariffs", methods=["GET", "POST"])
def tariffs():
    ensure_data_files()
    tariffs_by_year = load_tariffs()
    grid_powers = load_grid_power()

    if request.method == "POST":
        action = request.form.get("action", "tariff")
        if action == "tariff":
            year = int(request.form["year"])
            base_price_per_kw = float(request.form["base_price_per_kw"].replace(",", "."))
            contracted_power_kw_str = request.form["contracted_power_kw"].replace(",", ".")
            contracted_power_kw = float(contracted_power_kw_str) if contracted_power_kw_str else 0.0
            working_price_per_kwh = float(request.form["working_price_per_kwh"].replace(",", "."))
            additional_yearly_costs = float(request.form["additional_yearly_costs"].replace(",", "."))

            tariffs_by_year[year] = {
                "year": year,
                "base_price_per_kw": base_price_per_kw,
                "contracted_power_kw": contracted_power_kw,
                "working_price_per_kwh": working_price_per_kwh,
                "additional_yearly_costs": additional_yearly_costs,
            }

            save_tariffs(list(tariffs_by_year.values()))
            flash("Tariff saved.", "success")
        elif action == "grid_power":
            gp_date = request.form.get("gp_date")
            gp_power = request.form.get("gp_power", "").replace(",", ".")
            try:
                append_grid_power(gp_date, gp_power)
                flash("Grid connection power saved.", "success")
            except Exception as e:
                flash(f"Could not save grid connection power: {e}", "danger")

        return redirect(url_for("tariffs"))

    tariff_list = sorted(tariffs_by_year.values(), key=lambda t: t["year"])
    grid_powers_sorted = sorted(grid_powers, key=lambda g: g["date"])

    return render_template(
        "tariffs.html",
        tariffs=tariff_list,
        grid_powers=grid_powers_sorted,
        today=date.today().isoformat(),
    )


@app.route("/readings", methods=["GET", "POST"])
def readings_view():
    """
    List and edit all meter readings.
    """
    ensure_data_files()

    if request.method == "POST":
        action = request.form.get("action")
        readings = load_readings()

        try:
            idx = int(request.form.get("index", "-1"))
        except ValueError:
            idx = -1

        if 0 <= idx < len(readings):
            if action == "update":
                date_str = request.form.get("date")
                meter_str = request.form.get("meter_reading", "").replace(",", ".")
                try:
                    new_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    new_meter = float(meter_str)
                    readings[idx]["date"] = new_date
                    readings[idx]["meter_reading"] = new_meter
                    save_readings(readings)
                    flash("Reading updated.", "success")
                except Exception as e:
                    flash(f"Could not update reading: {e}", "danger")
            elif action == "delete":
                try:
                    del readings[idx]
                    save_readings(readings)
                    flash("Reading deleted.", "info")
                except Exception as e:
                    flash(f"Could not delete reading: {e}", "danger")
        else:
            flash("Invalid reading index.", "danger")

        return redirect(url_for("readings_view"))

    readings = load_readings()
    return render_template(
        "readings.html",
        readings=readings,
    )


if __name__ == "__main__":
    ensure_data_files()
    try:
        update_weather_history_if_needed()
    except Exception as e:
        logger.error(f"Initial weather history update failed: {e}")
    schedule_daily_weather_update()
    app.run(host="0.0.0.0", port=8080, debug=False)
