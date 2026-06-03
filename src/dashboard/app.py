"""
app.py
------
Aplicația Streamlit — interfața principală a proiectului.
Owner: Pricop Matei-Ioan
Input:  data/ctp_pulse.db + model/ctp_absa_bert/ + data/timetable/timetable.json
Output: UI web accesibil la localhost:8501
TODO:
- [ ] Navigare între 3 pagini: Review nou, Dashboard, Căutare per linie
- [ ] Pagina Review nou: input text, buton Submit, afișare aspecte extrase cu culori
- [ ] Dacă aspect punctualitate detected: afișare rezultat GPS check automat
- [ ] Pagina Dashboard: heatmap plotly linie x aspect (culoare = % recenzii negative)
- [ ] Pagina Dashboard: grafic trend temporal sentiment per aspect pe luni
- [ ] Salvare review nou + rezultate ABSA în SQLite după submit

streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from tab_delay import render_delay_tab

st.title('CTP Pulse Dashboard')


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


reviews, absa, stop_mentions = load_data()


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Probleme principale",
    "Stații problematice",
    "Heatmap stații",
    "Adaugă review",
    "Întârzieri"
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
    st.dataframe(top_aspects, width='stretch')


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

        st.dataframe(stop_summary, width='stretch')


with tab3:
    st.info(
        """
        Heatmap-ul reprezintă concentrarea stațiilor menționate în recenzii care conțin aspecte negative.
        
        Intensitatea culorilor nu indică trafic real, întârzieri măsurate sau date operaționale
        ale CTP Iași. Aceasta reflectă exclusiv densitatea problemelor raportate de utilizatori
        în recenziile analizate.
        """
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
            width='stretch'
        )
        
with tab4:
    st.header("Analiză Review")

    st.info(
        """
        Această secțiune va permite analizarea unei recenzii noi folosind
        modelul Aspect-Based Sentiment Analysis (ABSA).

        Funcționalitatea este în curs de implementare.
        """
    )

    review_text = st.text_area(
        "Introduceți textul recenziei",
        placeholder="Ex: Tramvaiele sunt foarte aglomerate și întârzie frecvent..."
    )

    st.button(
        "Analizează recenzia",
        disabled=True,
        width='stretch'
    )

    st.divider()

    st.subheader("Rezultate (exemplu demonstrativ)")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Sentiment general",
            "Negativ"
        )

    with col2:
        st.metric(
            "Aspecte detectate",
            3
        )

    st.dataframe(
        pd.DataFrame([
            {
                "Aspect": "punctualitate",
                "Sentiment": "negativ",
                "Fragment": "întârzie frecvent"
            },
            {
                "Aspect": "aglomeratie",
                "Sentiment": "negativ",
                "Fragment": "tramvaiele sunt foarte aglomerate"
            },
            {
                "Aspect": "confort_termic",
                "Sentiment": "neutru",
                "Fragment": "aerul condiționat funcționează uneori"
            }
        ]),
        width='stretch',
        hide_index=True
    )

    st.divider()

    st.subheader("Recenzii similare (exemplu)")

    st.caption("Vor fi afișate recenzii din corpus cu aspecte similare.")

    st.markdown("""
    **Review #102**

    „Autobuzele circulă cu întârziere și sunt foarte aglomerate la orele de vârf.”
    """)

    st.markdown("""
    **Review #245**

    „Tramvaiele sunt pline dimineața și timpul de așteptare este prea mare.”
    """)

with tab5:
    render_delay_tab()