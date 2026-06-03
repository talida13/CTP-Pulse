"""
delay_calculator.py
-------------------
Pentru o linie și o dată dată, calculează întârzierile față de orarul teoretic.
Owner: Pricop Matei-Ioan
Input:  data/processed/orar_ctp_all.json + data/ctp_pulse.db (tabel gps_snapshots)
Output: dict cu detalii per trecere detectată

Folosit din app.py și din analize standalone.
"""

import json
import math
import os
import re
import sqlite3
import unicodedata
import requests
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
TIMETABLE_PATH = Path("data/processed/orar_ctp_all.json")
DB_PATH        = Path("data/ctp_pulse.db")

# ── Constante ─────────────────────────────────────────────────────────────────
STOP_RADIUS_M = 60   # raza în metri pentru detectarea trecerii printr-o stație

API_KEY   = os.getenv("TRANZY_API_KEY", "A0rqhZpCABWjtmIp2WdtT4Pobn54A91d578sw90k")
AGENCY_ID = os.getenv("TRANZY_AGENCY_ID", "1")
TRANZY_HEADERS = {
    "X-API-KEY":   API_KEY,
    "X-Agency-Id": str(AGENCY_ID),
    "Accept":      "application/json",
}

# ── Mapare route_short_name → route_id_int (cache în memorie) ─────────────────
_route_id_cache: dict[str, int] = {}

def build_route_id_map() -> dict[str, int]:
    """
    Fetch /routes din Tranzy și construiește un dict:
        route_short_name_lower → route_id_int
    Ex: {"30b": 41, "42": 12, "3": 1, ...}
    Rezultatul e cache-uit în memorie — se apelează o singură dată per sesiune.
    """
    global _route_id_cache
    if _route_id_cache:
        return _route_id_cache
    try:
        resp = requests.get(
            "https://api.tranzy.ai/v1/opendata/routes",
            headers=TRANZY_HEADERS, timeout=10
        )
        resp.raise_for_status()
        for r in resp.json():
            short = str(r.get("route_short_name", "")).strip().lower()
            rid   = r.get("route_id")
            if short and rid is not None:
                _route_id_cache[short] = int(rid)
    except Exception as e:
        print(f"[delay_calculator] Nu s-a putut fetch /routes: {e}")
    return _route_id_cache


def resolve_route_id(route_key: str, timetable: dict) -> Optional[int]:
    """
    Rezolvă route_id-ul integer din Tranzy pentru o cheie din timetable.
    Strategie:
      1. route_short_name din /routes (ex: '30b' → 41)
      2. Fallback: parse numeric din route_key (ex: '42' → 42)
    """
    route_map = build_route_id_map()

    # Cheia din timetable (ex: "30b", "42") == route_short_name din Tranzy
    key_lower = route_key.strip().lower()
    if key_lower in route_map:
        return route_map[key_lower]

    # Fallback: dacă e pur numeric
    try:
        return int(route_key)
    except ValueError:
        pass

    # Fallback: extrage prefixul numeric
    m = re.match(r"(\d+)", route_key)
    if m:
        candidate = int(m.group(1))
        # verifică că există în map ca valoare
        if candidate in route_map.values():
            return candidate
        return candidate  # oricum încearcă

    return None


# ── Utilitare geospațiale ─────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distanța în metri între două coordonate GPS (formula Haversine)."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Normalizare nume stații ───────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase + elimină diacritice + spații multiple."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(ascii_text.lower().split())


# ── Tip zi din orar ───────────────────────────────────────────────────────────

def day_type(d: date) -> str:
    wd = d.weekday()
    if wd <= 4:  return "Luni-Vineri"
    if wd == 5:  return "Sambata"
    return "Duminica"


# ── Timetable ────────────────────────────────────────────────────────────────

def load_timetable() -> dict:
    with open(TIMETABLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_scheduled_times(timetable: dict, route_key: str, direction: str,
                        stop_name: str, day: date) -> list[datetime]:
    try:
        hours_map = timetable[route_key]["directions"][direction][stop_name][day_type(day)]
    except KeyError:
        return []
    times = []
    for hour_str, minutes in hours_map.items():
        for minute_str in minutes:
            times.append(datetime(day.year, day.month, day.day,
                                  int(hour_str), int(minute_str)))
    return sorted(times)


def list_stops(timetable: dict, route_key: str, direction: str) -> list[str]:
    try:
        return list(timetable[route_key]["directions"][direction].keys())
    except KeyError:
        return []


def list_directions(timetable: dict, route_key: str) -> list[str]:
    try:
        return list(timetable[route_key]["directions"].keys())
    except KeyError:
        return []


# ── Citire GPS din SQLite ─────────────────────────────────────────────────────

def load_gps_for_route(route_id: int, target_date: date,
                       until_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """
    Citește din DB doar rândurile relevante pentru ruta și ziua cerută.
    Returnează None dacă DB-ul nu există sau nu are date pentru ziua respectivă.
    """
    if not DB_PATH.exists():
        return None

    day_start = datetime(target_date.year, target_date.month, target_date.day).isoformat()
    day_end   = (until_time or datetime(target_date.year, target_date.month,
                                        target_date.day, 23, 59, 59)).isoformat()

    query = """
        SELECT collected_at, vehicle_id, latitude, longitude, speed
        FROM   gps_snapshots
        WHERE  route_id    = ?
          AND  collected_at >= ?
          AND  collected_at <= ?
        ORDER  BY collected_at
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        df = pd.read_sql_query(query, conn, params=(route_id, day_start, day_end),
                               parse_dates=["collected_at"])
        conn.close()
    except Exception:
        return None

    return df if not df.empty else None


def has_gps_data(target_date: date) -> bool:
    """Verifică rapid dacă există date GPS pentru ziua respectivă."""
    if not DB_PATH.exists():
        return False
    day_start = datetime(target_date.year, target_date.month, target_date.day).isoformat()
    day_end   = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59).isoformat()
    try:
        conn  = sqlite3.connect(DB_PATH)
        count = conn.execute(
            "SELECT COUNT(*) FROM gps_snapshots WHERE collected_at >= ? AND collected_at <= ?",
            (day_start, day_end)
        ).fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


# ── Detectare treceri ─────────────────────────────────────────────────────────

def detect_passages(gps_df: pd.DataFrame, stop_lat: float, stop_lon: float,
                    radius_m: float = STOP_RADIUS_M) -> list[datetime]:
    """
    Detectează momentele când un vehicul a trecut prin raza stației.
    Deduplicare: dacă același vehicul stă mai multe snapshot-uri în rază,
    se păstrează doar primul din fiecare vizită.
    """
    gps_df = gps_df.sort_values("collected_at").copy()
    gps_df["dist_m"] = gps_df.apply(
        lambda r: haversine_m(r["latitude"], r["longitude"], stop_lat, stop_lon),
        axis=1,
    )
    gps_df["in_radius"] = gps_df["dist_m"] <= radius_m

    passages: list[datetime] = []
    prev_in: dict[int, bool] = {}

    for _, row in gps_df.iterrows():
        vid         = row["vehicle_id"]
        currently   = row["in_radius"]
        was_in      = prev_in.get(vid, False)
        if currently and not was_in:
            passages.append(row["collected_at"].to_pydatetime())
        prev_in[vid] = currently

    return sorted(passages)


# ── Match trecere → cursă programată ─────────────────────────────────────────

def match_to_schedule(actual: datetime, scheduled: list[datetime],
                      window_min: int = 20) -> Optional[tuple[datetime, float]]:
    """
    Găsește cea mai apropiată cursă programată față de trecerea reală
    în fereastra ±window_min minute.
    Returnează (scheduled_time, delay_minutes) sau None.
    delay > 0 = întârziere, delay < 0 = înainte de program.
    """
    best, best_abs = None, float("inf")
    for sched in scheduled:
        diff = (actual - sched).total_seconds() / 60
        if abs(diff) <= window_min and abs(diff) < best_abs:
            best_abs = abs(diff)
            best     = (sched, round(diff, 1))
    return best


# ── Core ──────────────────────────────────────────────────────────────────────

def calculate_delays(
    route_key:   str,
    direction:   str,
    stop_name:   str,
    stop_lat:    float,
    stop_lon:    float,
    target_date: date,
    until_time:  Optional[datetime] = None,
) -> dict:
    """
    Calculează întârzierile pentru o rută+direcție+stație pe o zi dată,
    până la until_time (implicit: ora curentă).

    Returnează dict cu:
        route_key, direction, stop_name, date,
        passages: [{scheduled, actual, delay_min, status}],
        avg_delay_min, max_delay_min, num_passages,
        error (str dacă lipsesc date, altfel None)
    """
    result = dict(route_key=route_key, direction=direction, stop_name=stop_name,
                  date=target_date.isoformat(), passages=[],
                  avg_delay_min=None, max_delay_min=None, num_passages=0, error=None)

    # 1. Orar teoretic
    timetable = load_timetable()
    scheduled = get_scheduled_times(timetable, route_key, direction, stop_name, target_date)
    if not scheduled:
        result["error"] = f"Nu există orar pentru ruta {route_key} / {direction} / {stop_name}"
        return result

    if until_time:
        scheduled = [s for s in scheduled if s <= until_time]
    if not scheduled:
        result["error"] = "Nu există curse programate până la ora selectată."
        return result

    # 2. Rezolvă route_id integer din Tranzy via /routes (matching după route_short_name)
    route_id_int = resolve_route_id(route_key, timetable)
    if route_id_int is None:
        result["error"] = (
            f"Nu s-a putut determina route_id Tranzy pentru ruta '{route_key}'. "
            "Verifică că linia există în /routes."
        )
        return result

    # 3. GPS din SQLite — doar ruta și ziua relevantă
    gps_df = load_gps_for_route(route_id_int, target_date, until_time)
    if gps_df is None:
        result["error"] = (
            f"Nu există date GPS pentru {target_date.isoformat()}. "
            "Asigură-te că gps_collector.py rulează."
        )
        return result

    # 4. Detectare treceri
    passages_actual = detect_passages(gps_df, stop_lat, stop_lon)
    if not passages_actual:
        result["error"] = "Nu s-au detectat treceri GPS în raza stației pentru intervalul ales."
        return result

    # 5. Matching treceri → curse programate
    matched       = []
    used_scheduled = set()

    for actual in passages_actual:
        match = match_to_schedule(actual, scheduled)
        if match is None:
            continue
        sched_time, delay_min = match
        key = sched_time.isoformat()
        if key in used_scheduled:
            continue
        used_scheduled.add(key)
        matched.append({
            "scheduled": sched_time.strftime("%H:%M"),
            "actual":    actual.strftime("%H:%M"),
            "delay_min": delay_min,
            "status":    _delay_label(delay_min),
        })

    if not matched:
        result["error"] = "Nicio trecere GPS nu a putut fi asociată cu o cursă programată."
        return result

    delays = [m["delay_min"] for m in matched]
    result.update(
        passages     = matched,
        num_passages = len(matched),
        avg_delay_min = round(sum(delays) / len(delays), 1),
        max_delay_min = round(max(delays), 1),
    )
    return result


def _delay_label(delay_min: float) -> str:
    if delay_min < -1:   return "înainte"
    if delay_min <= 1:   return "la timp"
    if delay_min <= 5:   return "ușor întârziat"
    if delay_min <= 10:  return "întârziat"
    return "foarte întârziat"


# ── Helpers pentru Streamlit ──────────────────────────────────────────────────

def get_stops_with_coords(tranzy_stops: list[dict]) -> dict[str, tuple[float, float]]:
    return {normalize(s["stop_name"]): (s["stop_lat"], s["stop_lon"]) for s in tranzy_stops}


def find_stop_coords(stop_name_timetable: str,
                     stops_coords: dict[str, tuple[float, float]]) -> Optional[tuple[float, float]]:
    key = normalize(stop_name_timetable)
    if key in stops_coords:
        return stops_coords[key]
    for api_key, coords in stops_coords.items():
        if key in api_key or api_key in key:
            return coords
    return None


# ── CLI pentru testare rapidă ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) >= 4:
        res = calculate_delays(
            route_key   = sys.argv[1],
            direction   = sys.argv[2],
            stop_name   = sys.argv[3],
            stop_lat    = float(sys.argv[4]) if len(sys.argv) > 4 else 47.19052,
            stop_lon    = float(sys.argv[5]) if len(sys.argv) > 5 else 27.55848,
            target_date = date.today(),
        )
        pprint.pprint(res)
    else:
        print("Usage: python delay_calculator.py <route_key> <direction> <stop_name> [lat] [lon]")
        print("Ex:    python delay_calculator.py 42 'COPOU_TO_C.U.G. I' COPOU 47.19052 27.55848")