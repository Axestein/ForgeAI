"""
Tool registry implementations.
ForgeAI selects which tools to give the agent; these are the actual callables.
"""
from __future__ import annotations

import csv
import json
import math
import datetime
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


def csv_lookup(
    csv_path: str,
    column: str,
    value: str,
    return_columns: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Look up rows in a CSV where `column` matches `value`.
    Returns list of matching rows.
    """
    if not os.path.exists(csv_path):
        return {"error": f"CSV file not found: {csv_path}"}

    matches = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get(column, "")).strip().lower() == value.strip().lower():
                if return_columns:
                    matches.append({k: row[k] for k in return_columns if k in row})
                else:
                    matches.append(dict(row))

    return {"column": column, "value": value, "matches": matches, "count": len(matches)}


def calculator(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a math expression.
    Supports: +, -, *, /, **, sqrt, abs, round, floor, ceil.
    """
    # Whitelist safe names
    safe_names = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "pi": math.pi,
        "e": math.e,
        "log": math.log,
        "log10": math.log10,
        "pow": pow,
    }

    # Strip unsafe patterns
    cleaned = re.sub(r"[^0-9+\-*/().,\s\w]", "", expression)
    try:
        result = eval(cleaned, {"__builtins__": {}}, safe_names)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def weather_api(city: str) -> Dict[str, Any]:
    """
    Fetch current weather using open-meteo (no API key required).
    Uses geocoding + weather API.
    """
    try:
        import requests  # type: ignore

        # Step 1: Geocode
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_resp = requests.get(geo_url, timeout=5)
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return {"error": f"City not found: {city}"}

        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]

        # Step 2: Weather
        wx_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,wind_speed_10m,weathercode"
            f"&timezone=auto"
        )
        wx_resp = requests.get(wx_url, timeout=5)
        wx_data = wx_resp.json()
        current = wx_data.get("current", {})

        return {
            "city": loc.get("name", city),
            "country": loc.get("country", ""),
            "temperature_c": current.get("temperature_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": current.get("weathercode"),
        }
    except Exception as e:
        return {"error": str(e), "note": "Network not available in sandbox mode"}


def current_date() -> Dict[str, Any]:
    """Return the current UTC date and time."""
    now = datetime.datetime.utcnow()
    return {
        "utc_date": now.strftime("%Y-%m-%d"),
        "utc_time": now.strftime("%H:%M:%S"),
        "utc_datetime": now.isoformat(),
        "day_of_week": now.strftime("%A"),
    }


TOOL_FUNCTIONS = {
    "csv_lookup": csv_lookup,
    "calculator": calculator,
    "weather_api": weather_api,
    "current_date": current_date,
}


def call_tool(name: str, args: Dict[str, Any]) -> str:
    """Dispatch a tool call and return JSON string result."""
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOL_FUNCTIONS[name](**args)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
