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


tab1, tab2, tab3, tab4 = st.tabs([
    "Probleme principale",
    "Stații problematice",
    "Heatmap stații",
    "Adaugă review"
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
    st.dataframe(top_aspects, use_container_width=True)


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

        st.dataframe(stop_summary, use_container_width=True)


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
            use_container_width=True
        )
        
with tab4:
    st.subheader("Analiza unei recenzii noi")

    st.write(
        """
        Introduceți o recenzie pentru a observa aspectele și sentimentele
        identificate de modelul ABSA.
        """
    )

    review_text = st.text_area(
        "Textul recenziei",
        height=150,
        placeholder="Ex: Tramvaiele sunt foarte aglomerate și întârzie frecvent..."
    )