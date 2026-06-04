"""
tab_delay.py
------------
Tab-ul „Întârzieri GPS" pentru app.py.
Se importă și se apelează render_delay_tab(tab) din app.py.

Integrare în app.py:
    from src.dashboard.tab_delay import render_delay_tab
    tab5 = st.tabs([..., "Întârzieri GPS"])
    with tab5:
        render_delay_tab()
"""

import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
from pathlib import Path

# Importăm din același pachet
import sys
sys.path.append(str(Path(__file__).parent.parent))
from analysis.delay_calculator import (
    calculate_delays,
    load_timetable,
    list_directions,
    list_stops,
    get_stops_with_coords,
    find_stop_coords,
    day_type,
)

API_KEY   = os.getenv("TRANZY_API_KEY", "A0rqhZpCABWjtmIp2WdtT4Pobn54A91d578sw90k")
AGENCY_ID = os.getenv("TRANZY_AGENCY_ID", "1")
HEADERS   = {
    "X-API-KEY":   API_KEY,
    "X-Agency-Id": AGENCY_ID,
    "Accept":      "application/json",
}


@st.cache_data(ttl=3600)
def fetch_stops() -> list[dict]:
    """Fetch stații din Tranzy API (cache 1h)."""
    try:
        r = requests.get(
            "https://api.tranzy.ai/v1/opendata/stops",
            headers=HEADERS, timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Nu s-au putut încărca stațiile din API: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_live_vehicles() -> list[dict]:
    """Fetch vehicule live (cache 60s)."""
    try:
        r = requests.get(
            "https://api.tranzy.ai/v1/opendata/vehicles",
            headers=HEADERS, timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data
def load_timetable_cached() -> dict:
    return load_timetable()


def render_delay_tab():
    st.header("Analiză întârzieri GPS")

    st.info(
        "Selectează o linie, direcție, stație și intervalul de timp pentru care "
        "dorești să calculezi întârzierile față de orarul programat. "
    )

    timetable    = load_timetable_cached()
    tranzy_stops = fetch_stops()
    stops_coords = get_stops_with_coords(tranzy_stops) if tranzy_stops else {}

    # ── Selectoare ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        route_options = sorted(timetable.keys(), key=lambda x: int(x) if x.isdigit() else 999)
        route_labels  = {
            k: f"Linia {k} — {timetable[k]['route_name']} ({timetable[k]['vehicle_type']})"
            for k in route_options
        }
        selected_route = st.selectbox(
            "Linie",
            options=route_options,
            format_func=lambda k: route_labels[k],
        )

    with col2:
        directions = list_directions(timetable, selected_route)
        selected_dir = st.selectbox("Direcție", options=directions)

    stops = list_stops(timetable, selected_route, selected_dir)
    selected_stop = st.selectbox("Stație de referință", options=stops)

    col3, col4 = st.columns(2)
    with col3:
        selected_date = st.date_input("Data", value=date.today(), max_value=date.today())
    with col4:
        selected_time = st.time_input("Până la ora")

    until_dt = datetime.combine(selected_date, selected_time)

    # ── Buton calcul ──────────────────────────────────────────────────────────
    if st.button("Calculează întârzieri", type="primary", width='stretch'):
        with st.spinner("Se procesează datele GPS..."):

            # Rezolvă coordonatele stației
            coords = find_stop_coords(selected_stop, stops_coords)
            if coords is None:
                st.error(
                    f"Nu s-au găsit coordonatele pentru stația **{selected_stop}** în API-ul Tranzy. "
                    "Verifică că stația există în /stops."
                )
                return

            stop_lat, stop_lon = coords

            result = calculate_delays(
                route_key   = selected_route,
                direction   = selected_dir,
                stop_name   = selected_stop,
                stop_lat    = stop_lat,
                stop_lon    = stop_lon,
                target_date = selected_date,
                until_time  = until_dt,
            )

        # ── Eroare ────────────────────────────────────────────────────────────
        if result["error"]:
            st.warning(result["error"])

            # Dacă lipsesc date GPS, arătăm orarul teoretic
            if "GPS" in result["error"]:
                _show_theoretical_schedule(timetable, selected_route, selected_dir,
                                           selected_stop, selected_date, until_dt)
            return

        # ── Metrici sumar ─────────────────────────────────────────────────────
        st.subheader("Sumar")
        m1, m2, m3 = st.columns(3)
        m1.metric("Curse detectate",   result["num_passages"])
        m2.metric("Întârziere medie",  f"{result['avg_delay_min']} min")
        m3.metric("Întârziere maximă", f"{result['max_delay_min']} min")

        # ── Tabel detaliat ────────────────────────────────────────────────────
        st.subheader("Detalii per cursă")
        df = pd.DataFrame(result["passages"])
        if not df.empty:
            df.columns = ["Orar programat", "Trecere reală", "Întârziere (min)", "Status"]
            st.dataframe(
                df.style.map(_color_delay, subset=["Întârziere (min)"]),
                width="stretch",
                hide_index=True,
            )

        # ── Grafic ────────────────────────────────────────────────────────────
        st.subheader("Evoluție întârzieri în timp")
        _plot_delays(result["passages"])

        # ── Distribuție status ────────────────────────────────────────────────
        st.subheader("Distribuție status")
        _plot_status_pie(result["passages"])

    # ── Live preview (bonus) ───────────────────────────────────────────────────
    with st.expander("Preview live — vehicule active pe linia selectată"):
        vehicles = fetch_live_vehicles()
        try:
            route_id_int = int(timetable[selected_route]["route_id"])
        except (KeyError, ValueError):
            route_id_int = None

        if route_id_int:
            on_route = [v for v in vehicles if v.get("route_id") == route_id_int]
            if on_route:
                st.write(f"**{len(on_route)} vehicule** active acum pe linia {selected_route}:")
                df_live = pd.DataFrame([{
                    "ID": v.get("id"),
                    "Label": v.get("label"),
                    "Lat": v.get("latitude"),
                    "Lon": v.get("longitude"),
                    "Viteză (km/h)": v.get("speed"),
                    "Timestamp API": v.get("timestamp"),
                } for v in on_route])
                st.dataframe(df_live, width="stretch", hide_index=True)
            else:
                st.info(f"Niciun vehicul activ acum pe linia {selected_route}.")
        else:
            st.warning("Nu s-a putut determina route_id-ul din API.")


# ── Helpers vizuale ───────────────────────────────────────────────────────────

def _show_theoretical_schedule(timetable, route_key, direction, stop_name, target_date, until_dt):
    """Afișează orarul teoretic când nu avem date GPS."""
    from analysis.delay_calculator import get_scheduled_times
    scheduled = get_scheduled_times(timetable, route_key, direction, stop_name, target_date)
    scheduled = [s for s in scheduled if s <= until_dt]
    if scheduled:
        st.subheader(f"Orar teoretic pentru {stop_name} — {day_type(target_date)}")
        df = pd.DataFrame([{"Cursă programată": s.strftime("%H:%M")} for s in scheduled])
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("Nu există curse programate în intervalul selectat.")


def _plot_delays(passages: list[dict]):
    if not passages:
        return
    df = pd.DataFrame(passages)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["scheduled"],
        y=df["delay_min"],
        marker_color=[_delay_color(d) for d in df["delay_min"]],
        text=[f"{d:+.1f} min" for d in df["delay_min"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="la timp")
    fig.update_layout(
        xaxis_title="Cursă programată",
        yaxis_title="Întârziere (minute)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=350,
    )
    st.plotly_chart(fig, width='stretch')


def _plot_status_pie(passages: list[dict]):
    if not passages:
        return
    from collections import Counter
    counts = Counter(p["status"] for p in passages)
    color_map = {
        "înainte":           "#3b82f6",
        "la timp":           "#22c55e",
        "ușor întârziat":   "#f59e0b",
        "întârziat":        "#ef4444",
        "foarte întârziat": "#7f1d1d",
    }
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [color_map.get(l, "#9ca3af") for l in labels]
    fig = go.Figure(go.Pie(labels=labels, values=values, marker_colors=colors,
                           hole=0.4, textinfo="label+percent"))
    fig.update_layout(height=300, showlegend=True,
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width='stretch')


def _delay_color(delay: float) -> str:
    if delay < -1:    return "#3b82f6"
    if delay <= 1:    return "#22c55e"
    if delay <= 5:    return "#f59e0b"
    if delay <= 10:   return "#ef4444"
    return "#7f1d1d"


def _color_delay(val):
    """Pandas Styler pentru coloana întârziere."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    color = _delay_color(v)
    return f"color: {color}; font-weight: bold"