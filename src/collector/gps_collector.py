"""
gps_collector.py
----------------
Polling Tranzy API la fiecare 30s și salvează pozițiile vehiculelor CTP în SQLite.
Owner: Pricop Matei-Ioan
Input:  Tranzy API (TRANZY_API_KEY din .env)
Output: data/ctp_pulse.db → tabel gps_snapshots

Rulare:
    python src/collector/gps_collector.py
"""

import os
import time
import logging
import sqlite3
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY       = os.getenv("TRANZY_API_KEY", "A0rqhZpCABWjtmIp2WdtT4Pobn54A91d578sw90k")
AGENCY_ID     = int(os.getenv("TRANZY_AGENCY_ID", "1"))
VEHICLES_URL  = "https://api.tranzy.ai/v1/opendata/vehicles"
POLL_INTERVAL = 30  # secunde
DB_PATH       = Path("data/ctp_pulse.db")

HEADERS = {
    "X-API-KEY":   API_KEY,
    "X-Agency-Id": str(AGENCY_ID),
    "Accept":      "application/json",
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── DB ────────────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")   # permite citire concurentă din Streamlit
    conn.execute("PRAGMA synchronous=NORMAL") # mai rapid, tot sigur cu WAL
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Creează tabelul dacă nu există și adaugă index pe coloanele frecvent interogate."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gps_snapshots (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at  TEXT NOT NULL,        -- ISO8601, ora locală
            vehicle_id    INTEGER,
            label         TEXT,
            route_id      INTEGER,
            trip_id       TEXT,
            latitude      REAL,
            longitude     REAL,
            speed         REAL,
            vehicle_type  INTEGER,
            api_timestamp TEXT                  -- timestamp din răspunsul API
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_gps_route_collected
        ON gps_snapshots (route_id, collected_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_gps_collected
        ON gps_snapshots (collected_at)
    """)
    conn.commit()
    log.info("DB inițializată: %s", DB_PATH)


def insert_snapshot(conn: sqlite3.Connection, vehicles: list[dict], collected_at: datetime) -> int:
    """Inserează un snapshot complet într-o singură tranzacție. Returnează numărul de rânduri."""
    ts = collected_at.isoformat()
    rows = [
        (
            ts,
            v.get("id"),
            v.get("label"),
            v.get("route_id"),
            v.get("trip_id"),
            v.get("latitude"),
            v.get("longitude"),
            v.get("speed"),
            v.get("vehicle_type"),
            v.get("timestamp"),
        )
        for v in vehicles
    ]
    conn.executemany("""
        INSERT INTO gps_snapshots
            (collected_at, vehicle_id, label, route_id, trip_id,
             latitude, longitude, speed, vehicle_type, api_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    return len(rows)


# ── API ───────────────────────────────────────────────────────────────────────

def fetch_vehicles() -> list[dict]:
    """Fetch snapshot curent de la /vehicles. Returnează listă goală la eroare."""
    try:
        resp = requests.get(VEHICLES_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        log.warning("Timeout la /vehicles — skip snapshot")
    except requests.exceptions.HTTPError as e:
        log.warning("HTTP %s la /vehicles — %s", e.response.status_code, e)
    except requests.exceptions.RequestException as e:
        log.warning("Eroare rețea la /vehicles — %s", e)
    return []


# ── Main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    log.info("GPS Collector pornit | interval=%ds | agency=%d | db=%s",
             POLL_INTERVAL, AGENCY_ID, DB_PATH)

    conn = get_conn()
    init_db(conn)

    while True:
        now      = datetime.now()
        vehicles = fetch_vehicles()

        if vehicles:
            n = insert_snapshot(conn, vehicles, now)
            log.info("%s | %d vehicule salvate în DB", now.strftime("%H:%M:%S"), n)
        else:
            log.info("%s | Snapshot gol — nimic salvat", now.strftime("%H:%M:%S"))

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()