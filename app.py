import csv
import calendar
from pathlib import Path
from datetime import datetime, date

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
READINGS_CSV = DATA_DIR / "readings.csv"
TARIFFS_CSV = DATA_DIR / "tariffs.csv"
OFFSETS_CSV = DATA_DIR / "offsets.csv"
GRID_POWER_CSV = DATA_DIR / "grid_power.csv"

CSV_DATE_FORMAT = "%d.%m.%Y"  # e.g. 01.01.2022


# ---------- DATE HELPERS ----------

def parse_csv_date(s: str) -> date:
    """Parse a date from CSV (supports dd.mm.yyyy and yyyy-mm-dd)."""
    s = str(s).strip()
    # if something like "01.01.2022;46286.0" slipped in, split off anything after ;
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

            # Decide separator per line
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
                # Skip lines we can't parse
                continue

            rows.append({col1_name: dt, col2_name: val})

    rows.sort(key=lambda r: r[col1_name])
    return rows


# ---------- READINGS ----------

def load_readings():
    return _load_two_column_date_float_csv(READINGS_CSV, "date", "meter_reading")


def append_reading(date_str, meter_str):
    # date_str from HTML is yyyy-mm-dd
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    meter = float(meter_str.replace(",", "."))
    with READINGS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([format_csv_date(dt), meter])


# ---------- TARIFFS ----------

def load_tariffs():
    if not TARIFFS_CSV.exists():
        return {}
    # Tariffs are written by the app itself, so normal CSV is fine
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


# ---------- STATS & COSTS ----------

def compute_stats(readings, tariffs_by_year, offsets, grid_powers):
    """
    readings: physical meter readings
    offsets: list of offset events (for meter replacement)
    grid_powers: list of grid power changes (kW connected to grid)
    """
    virtual_readings = compute_virtual_readings(readings, offsets)
    if len(virtual_readings) < 2:
        return {
            "daily_entries": [],
            "all_time_avg_kwh_per_day": None,
            "all_time_avg_cost_per_day": None,
            "monthly_stats": [],
        }

    daily_entries = []
    grid_powers_sorted = sorted(grid_powers, key=lambda g: g["date"])

    for prev, curr in zip(virtual_readings, virtual_readings[1:]):
        days = (curr["date"] - prev["date"]).days
        if days <= 0:
            continue

        delta_kwh = curr["virtual_meter"] - prev["virtual_meter"]

        # Safety: if still negative, assume meter reset and use physical reading as delta
        if delta_kwh < 0:
            delta_kwh = curr["meter_reading"]

        daily_kwh = delta_kwh / days
        year = curr["date"].year
        tariff = tariffs_by_year.get(year)
        daily_cost = None

        if tariff:
            days_in_year = 366 if calendar.isleap(year) else 365

            # Use the correct kW for that date
            power_kw = get_power_kw_for_date(curr["date"], grid_powers_sorted, tariffs_by_year)

            yearly_fixed = (tariff["base_price_per_kw"] * power_kw
                            + tariff["additional_yearly_costs"])
            fixed_per_day = yearly_fixed / days_in_year

            energy_cost = daily_kwh * tariff["working_price_per_kwh"]
            daily_cost = fixed_per_day + energy_cost

        daily_entries.append({
            "date": curr["date"],
            "daily_kwh": daily_kwh,
            "daily_cost": daily_cost,
        })

    if not daily_entries:
        return {
            "daily_entries": [],
            "all_time_avg_kwh_per_day": None,
            "all_time_avg_cost_per_day": None,
            "monthly_stats": [],
        }

    total_kwh = sum(e["daily_kwh"] for e in daily_entries)
    all_time_avg_kwh = total_kwh / len(daily_entries)

    cost_entries = [e["daily_cost"] for e in daily_entries if e["daily_cost"] is not None]
    all_time_avg_cost = (sum(cost_entries) / len(cost_entries)) if cost_entries else None

    monthly_groups = {}
    for e in daily_entries:
        key = e["date"].strftime("%Y-%m")
        if key not in monthly_groups:
            monthly_groups[key] = {"total_kwh": 0.0, "total_cost": 0.0, "days": 0}
        monthly_groups[key]["total_kwh"] += e["daily_kwh"]
        monthly_groups[key]["days"] += 1
        if e["daily_cost"] is not None:
            monthly_groups[key]["total_cost"] += e["daily_cost"]

    monthly_stats = []
    for month_key in sorted(monthly_groups.keys()):
        group = monthly_groups[month_key]
        avg_kwh = group["total_kwh"] / group["days"]
        avg_cost = (group["total_cost"] / group["days"]) if group["total_cost"] else None
        monthly_stats.append({
            "month": month_key,
            "avg_kwh_per_day": avg_kwh,
            "avg_cost_per_day": avg_cost,
        })

    return {
        "daily_entries": daily_entries,
        "all_time_avg_kwh_per_day": all_time_avg_kwh,
        "all_time_avg_cost_per_day": all_time_avg_cost,
        "monthly_stats": monthly_stats,
    }


# ---------- ROUTES ----------

@app.route("/", methods=["GET", "POST"])
def index():
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

        return redirect(url_for("index"))

    readings = load_readings()
    tariffs = load_tariffs()
    offsets = load_offsets()
    grid_powers = load_grid_power()
    stats = compute_stats(readings, tariffs, offsets, grid_powers)

    daily_entries = stats["daily_entries"][-60:]
    daily_labels = [e["date"].isoformat() for e in daily_entries]
    daily_values = [round(e["daily_kwh"], 2) for e in daily_entries]

    monthly_labels = [m["month"] for m in stats["monthly_stats"]]
    monthly_values = [round(m["avg_kwh_per_day"], 2) for m in stats["monthly_stats"]]

    return render_template(
        "index.html",
        readings=readings[-20:],
        stats=stats,
        today=date.today().isoformat(),
        daily_labels=daily_labels,
        daily_values=daily_values,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        offsets=offsets,
        grid_powers=grid_powers,
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

if __name__ == "__main__":
    # In Docker: bind to all interfaces, disable debug to avoid extra restrictions
    app.run(host="0.0.0.0", port=5000, debug=False)
