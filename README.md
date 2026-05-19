# CTP Pulse – Analiza experienței utilizatorilor în transportul public din Iași

> Aspect-Based Sentiment Analysis pe recenzii CTP Iași, validat cu date GTFS real-time de la tranzy.ai

## Descriere

CTP Pulse combină **date obiective** (pozițiile autobuzelor și respectarea programului din API-ul tranzy.ai) cu **opiniile subiective** ale călătorilor (recenzii Google Maps, Facebook) pentru a identifica exact ce nemulțumiri sunt confirmate de realitate și ce aspecte ale transportului public ieșean necesită îmbunătățire.

**Întrebarea de cercetare:** *Oamenii se plâng de întârzieri — datele GTFS confirmă sau infirmă?*

## Echipă

| Nume | Rol principal |
|------|--------------|
| Afloroaiei Andrei-Gabriel | 
| Caraman Talida | 
| Postolache Andrei | 
| Pricop Matei-Ioan | 


## Aspecte analizate (ABSA targets)

| # | Aspect | Cuvinte cheie tipice |
|---|--------|---------------------|
| 1 | Punctualitate / întârzieri | întârzie, nu vine, mereu târziu |
| 2 | Aglomerație | plin, înghesuială, nu încap |
| 3 | Curățenie vehicul | murdar, mizerie, miros |
| 4 | Comportament șofer | șofer, nepoliticos, brusc |
| 5 | Aer condiționat / căldură | cald, frig, aer, climatizare |
| 6 | Validare bilet / e-ticketing | card, validare, aparat, plată |
| 7 | Informații stație / panouri | panou, orar, informații, anunț |
| 8 | Frecvența curselor | rar, des, o dată la, frecvență |
| 9 | Siguranță nocturnă | noaptea, întunecos, nesigur |
| 10 | Accesibilitate persoane cu dizabilități | cărucior, rampa, accesibil |

## Surse de date

- **tranzy.ai/opendata** — GTFS real-time (poziții vehicule, orare, stații), actualizat la 20s
- **Google Maps Places API** — recenzii stații CTP, pagina CTP Iași
- **Facebook CTP Iași** — comentarii publice postări oficiale
- **App Store / Google Play** — recenzii aplicația tranzy.app

## Structura proiectului

```
ctp-pulse/
├── data/
│   ├── raw/          # date brute colectate, neatinse
│   ├── processed/    # date curățate, gata de modelare
│   └── gtfs/         # snapshot-uri GTFS și calcule întârzieri
├── src/
│   ├── collection/   # scripturi colectare recenzii + GTFS polling
│   ├── preprocessing/# curățare text, normalizare diacritice
│   ├── absa/         # extracție aspecte + clasificare sentiment
│   ├── validation/   # corelare sentiment ↔ date GTFS
│   └── dashboard/    # interfață Streamlit
├── notebooks/        # explorare, EDA, experimente
├── tests/            # unit tests per modul
├── reports/          # rapoarte intermediare + raport civic final
├── docs/             # documentație tehnică detaliată
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalare

```bash
git clone https://github.com/<org>/ctp-pulse.git
cd ctp-pulse
pip install -r requirements.txt
```

## Rulare rapidă

```bash
# Colectare recenzii
python src/collection/collect_reviews.py

# Polling GTFS (rulează continuu)
python src/collection/gtfs_poller.py

# Pipeline ABSA complet
python src/absa/run_pipeline.py

# Dashboard
streamlit run src/dashboard/app.py
```

## Tehnologii

`Python 3.10+` · `spaCy ro_core_news_md` · `HuggingFace Transformers` · `BERT-ro` · `gtfs-realtime-bindings` · `Streamlit` · `folium` · `plotly` · `pandas` · `scikit-learn`

## Licență

MIT
