import os
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import date, datetime
from tab_delay import render_delay_tab

from src.nlp.predict import PredictionError, load_pipeline, predict_review
from src.analysis.delay_calculator import (
    calculate_delays,
    load_timetable,
    list_directions,
    list_stops,
    get_stops_with_coords,
    find_stop_coords,
)

@st.cache_data
def load_timetable_cached() -> dict:
    return load_timetable()

st.title("CTP Pulse Dashboard")

IASI_LAT = 47.1585
IASI_LON = 27.6014

@st.cache_data
def load_data():
    reviews = pd.read_csv("data/processed/reviews_merged.csv")
    absa = pd.read_csv("data/processed/absa_flat_dataset.csv")
    stop_mentions = pd.read_csv("data/processed/review_stop_mentions.csv")

    reviews["review_id"] = reviews["review_id"].astype(str)
    absa["review_id"] = absa["review_id"].astype(str)
    stop_mentions["review_id"] = stop_mentions["review_id"].astype(str)

    return reviews, absa, stop_mentions

@st.cache_data
def load_evaluation_data():
    eval_df = pd.read_csv("data/processed/evaluation_results.csv")
    pipeline_preds = pd.read_csv("data/processed/pipeline_test_predictions.csv")
    aspect_preds = pd.read_csv("data/processed/aspect_test_predictions.csv")
    sentiment_preds = pd.read_csv("data/processed/sentiment_test_predictions.csv")

    return eval_df, pipeline_preds, aspect_preds, sentiment_preds

eval_df, pipeline_preds, aspect_preds, sentiment_preds = load_evaluation_data()

def metric_value(module, method, scope, metric):
    row = eval_df[
        (eval_df["module"] == module)
        & (eval_df["method"] == method)
        & (eval_df["scope"] == scope)
        & (eval_df["metric"] == metric)
    ]

    if row.empty:
        return None

    return float(row.iloc[0]["value"])

def fmt(value):
    if value is None:
        return "—"
    return f"{value:.4f}"

@st.cache_resource
def load_absa_pipeline():
    return load_pipeline()

reviews, absa, stop_mentions = load_data()

def _run_punctuality_check(text: str, target_date: date) -> None:
    import re

    patterns = [
        r"\blinia\s+([0-9]{1,3}[a-zA-Z]?)\b",
        r"\bautobuzul?\s+([0-9]{1,3}[a-zA-Z]?)\b",
        r"\btramvaiul?\s+([0-9]{1,3}[a-zA-Z]?)\b",
        r"\btroleibuzul?\s+([0-9]{1,3}[a-zA-Z]?)\b",
        r"\bnr\.?\s*([0-9]{1,3}[a-zA-Z]?)\b",
    ]
    found_routes = []
    seen = set()
    text_low = text.lower()
    for pat in patterns:
        for m in re.finditer(pat, text_low):
            key = m.group(1).lower()
            if key not in seen:
                seen.add(key)
                found_routes.append(key)

    timetable = load_timetable_cached()

    valid_routes = [k for k in found_routes if k in timetable]

    if not valid_routes:
        if found_routes:
            st.warning(
                f"Am identificat posibila linie **{', '.join(found_routes)}** în text, "
                "dar nu există în orarul CTP. Verificați numărul liniei."
            )
        else:
            st.warning(
                "Nu am identificat nicio linie CTP în textul recenziei. "
                "Menționați numărul liniei (ex: *linia 42*) pentru verificare."
            )
        return

    tranzy_stops = fetch_stops_cached()
    stops_coords = get_stops_with_coords(tranzy_stops) if tranzy_stops else {}

    route_key  = valid_routes[0]
    directions = list_directions(timetable, route_key)
    if not directions:
        st.warning(f"Nu există direcții în orar pentru linia {route_key}.")
        return

    direction = directions[0]
    stops     = list_stops(timetable, route_key, direction)

    stop_name = stop_lat = stop_lon = None
    for candidate in stops[1:4] if len(stops) > 2 else stops:
        coords = find_stop_coords(candidate, stops_coords)
        if coords:
            stop_name = candidate
            stop_lat, stop_lon = coords
            break

    if stop_name is None:
        st.warning(
            f"Nu s-au găsit coordonate GPS pentru stațiile liniei {route_key}. "
            "Verificați că Tranzy /stops returnează stațiile liniei."
        )
        return

    with st.spinner(f"Se calculează întârzierile pentru linia {route_key}..."):
        result = calculate_delays(
            route_key   = route_key,
            direction   = direction,
            stop_name   = stop_name,
            stop_lat    = stop_lat,
            stop_lon    = stop_lon,
            target_date = target_date,
        )

    if result["error"]:
        st.warning(f"⚠️ {result['error']}")
        return

    passages  = result.get("passages", [])
    if not passages:
        st.info("Nu s-au detectat treceri GPS pentru ziua selectată.")
        return

    delays      = [p["delay_min"] for p in passages]
    avg_delay   = round(sum(delays) / len(delays), 1)
    max_delay   = round(max(delays), 1)
    num_delayed = sum(1 for d in delays if d > 2)
    pct_delayed = round(100 * num_delayed / len(delays), 1)

    if pct_delayed >= 40:
        st.error(
            f"✅ **Confirmat de GPS** — {pct_delayed}% dintre curse au fost întârziate "
            f"pe linia {route_key} în ziua recenziei "
            f"(medie **{avg_delay} min**, max **{max_delay} min**)."
        )
    elif pct_delayed <= 15:
        st.success(
            f"❌ **Neconfirmat de GPS** — doar {pct_delayed}% dintre curse au depășit "
            f"2 minute întârziere pe linia {route_key} "
            f"(medie **{avg_delay} min**)."
        )
    else:
        st.warning(
            f"⚠️ **Situație mixtă** — {pct_delayed}% curse întârziate pe linia {route_key} "
            f"(medie **{avg_delay} min**, max **{max_delay} min**). "
            "Nu se poate trage o concluzie clară."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linie analizată",   route_key)
    c2.metric("Curse analizate",   len(delays))
    c3.metric("Curse întârziate",  f"{num_delayed} ({pct_delayed}%)")
    c4.metric("Întârziere medie",  f"{avg_delay} min")

    with st.expander("Detalii curse", expanded=False):
        _STATUS_EMOJI = {
            "înainte":          "🔵",
            "la timp":          "🟢",
            "ușor întârziat":   "🟡",
            "întârziat":        "🟠",
            "foarte întârziat": "🔴",
        }
        df = pd.DataFrame([
            {
                "Programat":       p["scheduled"],
                "Efectiv":         p["actual"],
                "Diferență (min)": p["delay_min"],
                "Status":          f"{_STATUS_EMOJI.get(p['status'], '')} {p['status']}",
            }
            for p in passages
        ])
        st.dataframe(df, width='stretch', hide_index=True)

@st.cache_data(ttl=3600)
def fetch_stops_cached() -> list[dict]:
    import requests
    try:
        r = requests.get(
            "https://api.tranzy.ai/v1/opendata/stops",
            headers={
                "X-API-KEY":   os.getenv("TRANZY_API_KEY", "A0rqhZpCABWjtmIp2WdtT4Pobn54A91d578sw90k"),
                "X-Agency-Id": os.getenv("TRANZY_AGENCY_ID", "1"),
                "Accept":      "application/json",
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Probleme principale",
    "Stații problematice",
    "Heatmap stații",
    "Adaugă review",
    "Întârzieri",
    "Evaluare sistem"
])

with tab1:
    st.subheader("Top aspecte negative")

    negative_absa = absa[absa["sentiment"] == "negativ"]

    top_aspects = (
        negative_absa["aspect"]
        .value_counts()
        .reset_index()
    )
    top_aspects.columns = ["aspect", "count"]

    st.bar_chart(top_aspects.set_index("aspect"))
    st.dataframe(top_aspects, width="stretch")

with tab2:
    st.subheader("Stații cu cele mai multe probleme raportate")

    negative_absa = absa[absa["sentiment"] == "negativ"].copy()

    stop_issues = stop_mentions.merge(
        negative_absa[["review_id", "aspect", "sentiment", "fragment"]],
        on="review_id",
        how="inner"
    )

    if stop_issues.empty:
        st.info("Nu au fost găsite stații asociate cu aspecte negative.")
    else:
        stop_summary = (
            stop_issues
            .groupby(["stop_name", "lat", "lon"])
            .agg(
                negative_mentions=("aspect", "count"),
                reviews=("review_id", "nunique"),
                top_aspects=("aspect", lambda x: ", ".join(x.value_counts().head(3).index))
            )
            .reset_index()
            .sort_values("negative_mentions", ascending=False)
        )

        st.bar_chart(
            stop_summary.set_index("stop_name")["negative_mentions"]
        )

        st.dataframe(stop_summary, width="stretch")

with tab3:
    st.info(
        "Heatmap-ul reprezintă concentrarea stațiilor menționate în recenzii care conțin aspecte negative.\n\n"
        "Intensitatea culorilor nu indică trafic real, întârzieri măsurate sau date operaționale\n"
        "ale CTP Iași. Aceasta reflectă exclusiv densitatea problemelor raportate de utilizatori\n"
        "în recenziile analizate."
    )
    st.subheader("Heatmap pe stații menționate în review-uri negative")

    negative_absa = absa[absa["sentiment"] == "negativ"].copy()

    stop_issues = stop_mentions.merge(
        negative_absa[["review_id", "aspect", "sentiment", "fragment"]],
        on="review_id",
        how="inner"
    )

    if stop_issues.empty:
        st.info("Nu există date suficiente pentru heatmap.")
    else:
        heat_df = (
            stop_issues
            .groupby(["stop_name", "lat", "lon"])
            .agg(
                weight=("aspect", "count"),
                reviews=("review_id", "nunique"),
                top_aspects=("aspect", lambda x: ", ".join(x.value_counts().head(3).index))
            )
            .reset_index()
        )

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=heat_df,
            get_position="[lon, lat]",
            get_weight="weight",
            radiusPixels=70,
            intensity=1,
            threshold=0.05
        )

        points_layer = pdk.Layer(
            "ScatterplotLayer",
            data=heat_df,
            get_position="[lon, lat]",
            get_radius=90,
            get_fill_color=[255, 80, 80, 150],
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=IASI_LAT,
            longitude=IASI_LON,
            zoom=12,
            pitch=35
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[heatmap_layer, points_layer],
                initial_view_state=view_state,
                tooltip={
                    "text": (
                        "Stație: {stop_name}\n"
                        "Mențiuni negative: {weight}\n"
                        "Review-uri: {reviews}\n"
                        "Aspecte: {top_aspects}"
                    )
                }
            )
        )

        st.subheader("Date pentru hartă")
        st.dataframe(
            heat_df.sort_values("weight", ascending=False),
            width="stretch"
        )

with tab4:
    st.header("Analiză Review")

    st.info(
        "Această secțiune analizează o recenzie nouă folosind\n"
        "modelul Aspect-Based Sentiment Analysis (ABSA).\n"
        "Dacă este detectat aspectul **punctualitate**, puteți verifica\n"
        "automat dacă întârzierile sunt confirmate de datele GPS Tranzy\n"
        "din ziua în care a fost lăsată recenzia."
    )

    review_text = st.text_area(
        "Introduceți textul recenziei",
        placeholder="Ex: Tramvaiele sunt foarte aglomerate și întârzie frecvent...",
    )

    review_date_input = st.date_input(
        "Data recenziei",
        value=date.today(),
        max_value=date.today(),
        help=(
            "Necesară pentru verificarea GPS a punctualității. "
            "GPS Collector trebuie să fi rulat în ziua respectivă."
        ),
    )

    analyze_clicked = st.button(
        "Analizează recenzia",
        disabled=not review_text.strip(),
        width='stretch',
    )

    if analyze_clicked:
        try:
            with st.spinner("Se analizează recenzia..."):
                pipeline = load_absa_pipeline()
                result = predict_review(
                    text=review_text,
                    pipeline=pipeline,
                   # review_date=review_date_input,
                )

            aspects_result = result["aspects"]

            st.divider()
            st.subheader("Rezultate")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Sentiment general",
                    str(result["overall_sentiment"]).capitalize(),
                )

            with col2:
                st.metric(
                    "Scor general",
                    fmt(float(result["overall_score"])),
                )

            with col3:
                st.metric(
                    "Aspecte detectate",
                    len(aspects_result),
                )

            if not aspects_result:
                st.warning("Nu au fost detectate aspecte peste pragul modelului.")
            else:
                result_df = pd.DataFrame([
                    {
                        "Aspect": item["aspect"],
                        "Sentiment": item["sentiment"],
                        "Scor sentiment": round(float(item["score"]), 4),
                        "Scor aspect": round(float(item["aspect_score"]), 4),
                        "Fragment": item["fragment"],
                    }
                    for item in aspects_result
                ])

                st.dataframe(
                    result_df,
                    width='stretch',
                    hide_index=True,
                )

            has_punctuality = any(
                a.get("aspect") == "punctualitate" for a in aspects_result
            )

            if has_punctuality:
                st.divider()
                st.subheader("🚌 Verificare punctualitate GPS (Calcul Automat)")
                _run_punctuality_check(review_text, review_date_input)

        except PredictionError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analiza review-ului a eșuat: {exc}")

with tab5:
    render_delay_tab()

with tab6:
    st.header("Evaluarea sistemului ABSA")

    st.info(
        "Această secțiune prezintă evaluarea sistemului ABSA pe trei niveluri:\n"
        "detectorul de aspecte, clasificatorul de sentiment și pipeline-ul end-to-end."
    )

    st.subheader("Performanță pe module")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model 1 — Aspect detector")

        st.metric(
            "Accuracy",
            fmt(metric_value("aspect_detector", "BERT_pair", "overall", "accuracy"))
        )
        st.metric(
            "Precision",
            fmt(metric_value("aspect_detector", "BERT_pair", "overall", "precision"))
        )
        st.metric(
            "Recall",
            fmt(metric_value("aspect_detector", "BERT_pair", "overall", "recall"))
        )
        st.metric(
            "F1",
            fmt(metric_value("aspect_detector", "BERT_pair", "overall", "f1"))
        )

    with col2:
        st.markdown("### Model 2 — Sentiment classifier")

        st.metric(
            "Accuracy",
            fmt(metric_value("sentiment_classifier", "BERT_pair", "overall", "accuracy"))
        )
        st.metric(
            "Macro F1",
            fmt(metric_value("sentiment_classifier", "BERT_pair", "overall", "f1_macro"))
        )
        st.metric(
            "Weighted F1",
            fmt(metric_value("sentiment_classifier", "BERT_pair", "overall", "f1_weighted"))
        )

    st.warning(
        "Observație: bottleneck-ul sistemului este detectorul de aspecte.\n"
        "Modelul de sentiment are performanță ridicată, dar pipeline-ul final este limitat\n"
        "de aspectele ratate de primul model."
    )

    st.divider()

    st.subheader("Comparație BERT vs baseline")

    comparison = eval_df[
        (eval_df["module"] == "pipeline")
        & (eval_df["scope"] == "end_to_end")
        & (eval_df["metric"].isin(["aspect_f1", "tuple_f1"]))
    ].copy()

    comparison["method"] = comparison["method"].replace({
        "BERT_pair_pipeline": "BERT",
        "keyword_lexicon_baseline": "Baseline"
    })

    chart_df = comparison.pivot(
        index="metric",
        columns="method",
        values="value"
    )

    st.bar_chart(chart_df)

    st.dataframe(
        comparison[["method", "metric", "value"]],
        width='stretch',
        hide_index=True,
    )

    st.success(
        "Pipeline-ul BERT depășește baseline-ul lexical, mai ales pentru tuple_f1,\n"
        "adică pentru predicția completă aspect + sentiment."
    )

    st.divider()

    st.subheader("Performanță pe fiecare aspect")

    aspect_f1 = eval_df[
        (eval_df["module"] == "aspect_detector")
        & (eval_df["method"] == "BERT_pair")
        & (eval_df["metric"] == "f1")
        & (eval_df["scope"] != "overall")
    ].copy()

    aspect_f1 = aspect_f1.sort_values("value", ascending=False)

    st.bar_chart(aspect_f1.set_index("scope")["value"])

    st.dataframe(
        aspect_f1.rename(columns={
            "scope": "aspect",
            "value": "f1"
        })[["aspect", "f1"]],
        width='stretch',
        hide_index=True,
    )

    st.divider()

    st.subheader("Exemple de predicții end-to-end")

    st.caption(
        "Tabelul compară perechile reale aspect-sentiment cu predicțiile BERT\n"
        "și cu predicțiile baseline-ului lexical."
    )

    examples = pipeline_preds[
        [
            "review_id",
            "text",
            "gold_tuples",
            "pred_tuples_bert",
            "pred_tuples_baseline",
        ]
    ].copy()

    st.dataframe(
        examples,
        width='stretch',
        hide_index=True,
    )

    st.divider()

    st.subheader("Exemple unde BERT greșește")

    errors = pipeline_preds[
        pipeline_preds["gold_tuples"] != pipeline_preds["pred_tuples_bert"]
    ][
        [
            "review_id",
            "text",
            "gold_tuples",
            "pred_tuples_bert",
            "pred_tuples_baseline",
        ]
    ]

    st.dataframe(
        errors.head(10),
        width='stretch',
        hide_index=True,
    )