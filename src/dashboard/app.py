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


reviews, absa, stop_mentions = load_data()


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Probleme principale",
    "Stații problematice",
    "Heatmap stații",
    "Adaugă review",
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
        use_container_width=True
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
        use_container_width=True,
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
    st.header("Evaluarea sistemului ABSA")

    st.info(
        """
        Această secțiune prezintă evaluarea sistemului ABSA pe trei niveluri:
        detectorul de aspecte, clasificatorul de sentiment și pipeline-ul end-to-end.
        """
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
        """
        Observație: bottleneck-ul sistemului este detectorul de aspecte.
        Modelul de sentiment are performanță ridicată, dar pipeline-ul final este limitat
        de aspectele ratate de primul model.
        """
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
        use_container_width=True,
        hide_index=True
    )

    st.success(
        """
        Pipeline-ul BERT depășește baseline-ul lexical, mai ales pentru tuple_f1,
        adică pentru predicția completă aspect + sentiment.
        """
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
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Exemple de predicții end-to-end")

    st.caption(
        """
        Tabelul compară perechile reale aspect-sentiment cu predicțiile BERT
        și cu predicțiile baseline-ului lexical.
        """
    )

    examples = pipeline_preds[
        [
            "review_id",
            "text",
            "gold_tuples",
            "pred_tuples_bert",
            "pred_tuples_baseline"
        ]
    ].copy()

    st.dataframe(
        examples,
        use_container_width=True,
        hide_index=True
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
            "pred_tuples_baseline"
        ]
    ]

    st.dataframe(
        errors.head(10),
        use_container_width=True,
        hide_index=True
    )

