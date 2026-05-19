# Plan de proiect — CTP Pulse

## Echipă și responsabilități

Toți membrii echipei scriu cod și documentație. Responsabilitățile de mai jos indică **ownership-ul principal** per modul, nu munca exclusivă.

---

### Afloroaiei Andrei-Gabriel — Colectare date & GTFS pipeline
**Modul principal:** `src/collection/`

Responsabilități:
- Configurare acces Google Maps Places API + testare cotă gratuită
- Script colectare recenzii Google Maps pentru stațiile CTP (collect_reviews.py)
- Script scraping comentarii Facebook CTP Iași (collect_facebook.py)
- Script colectare recenzii App Store / Google Play pentru tranzy.app
- Script polling GTFS real-time tranzy.ai (gtfs_poller.py) — rulează continuu 30s
- Calcul întârzieri per linie din datele GPS (oră reală − oră planificată)
- Documentare format date colectate în `docs/data_sources.md`

Livrabile proprii:
- `src/collection/collect_reviews.py`
- `src/collection/collect_facebook.py`
- `src/collection/gtfs_poller.py`
- `src/collection/compute_delays.py`
- `docs/data_sources.md`
- `tests/test_collection.py`

---

### Caraman Talida — Preprocessing & Aspect Extraction
**Modul principal:** `src/preprocessing/` + `src/absa/aspect_extractor.py`

Responsabilități:
- Pipeline curățare text românesc (lowercase, diacritice, emoji → token, deduplicare)
- Normalizare diacritice lipsă (ro-diacritics / unidecode fallback)
- Definire și rafinare listă aspecte CTP (10 aspecte țintă)
- Extracție aspecte cu spaCy ro_core_news_md (noun chunks + dicționar domeniu)
- Construire dicționar de cuvinte cheie per aspect
- Adnotare manuală subset 200 exemple pentru training/test set
- Documentare în `docs/aspects.md`

Livrabile proprii:
- `src/preprocessing/cleaner.py`
- `src/preprocessing/normalizer.py`
- `src/absa/aspect_extractor.py`
- `src/absa/aspect_lexicon.py` (dicționar cuvinte cheie)
- `data/processed/annotated_sample.csv` (200 exemple adnotate)
- `docs/aspects.md`
- `tests/test_preprocessing.py`

---

### Postolache Andrei — Modelare sentiment & Evaluare NLP
**Modul principal:** `src/absa/sentiment_classifier.py` + `src/absa/evaluator.py`

Responsabilități:
- Implementare baseline lexical (SentiWordNet adaptat pentru română)
- Fine-tuning BERT-ro (readerbench/bert-base-romanian) pe subset adnotat
- Clasificare polaritate per aspect: POZITIV / NEGATIV / NEUTRU
- Calcul metrici: Precision, Recall, F1 per aspect și per model
- Calcul inter-annotator agreement (Cohen's Kappa) pe setul adnotat
- Comparație baseline vs. BERT-ro — tabel rezultate
- Documentare în `docs/model.md`

Livrabile proprii:
- `src/absa/sentiment_classifier.py`
- `src/absa/baseline_lexical.py`
- `src/absa/evaluator.py`
- `notebooks/03_model_training.ipynb`
- `notebooks/04_evaluation.ipynb`
- `reports/model_results.md`
- `docs/model.md`
- `tests/test_absa.py`

---

### Pricop Matei-Ioan — Validare civică & Dashboard
**Modul principal:** `src/validation/` + `src/dashboard/`

Responsabilități:
- Corelare sentiment negativ per aspect ↔ întârziere medie per linie (Pearson/Spearman)
- Analiză per interval orar (dimineață / prânz / seară / noapte)
- Identificare linii cu discrepanță mare (oamenii se plâng dar datele sunt ok, și invers)
- Dashboard Streamlit: hartă folium + heatmap linie × aspect
- Filtre interactive: linie, aspect, interval orar
- Raport civic final (1–2 pagini, non-tehnic, pentru Primăria Iași)
- Slide deck prezentare finală

Livrabile proprii:
- `src/validation/correlator.py`
- `src/validation/temporal_analysis.py`
- `src/dashboard/app.py`
- `src/dashboard/components/` (hartă, heatmap, filtre)
- `reports/civic_report.pdf`
- `docs/dashboard.md`
- `tests/test_validation.py`

---

## Milestones detaliate

### Săptămâna 1 — Setup & primele date
- [ ] Repo inițializat pe GitHub, toți membrii au acces
- [ ] `.env` completat cu cheile API (Google Maps, tranzy.ai)
- [ ] Primele 50–100 recenzii colectate și salvate în `data/raw/`
- [ ] Script GTFS pornit și loghează date
- [ ] Canva cu cele 4 dimensiuni finalizat și predat

### Săptămâna 2 — Date curate + aspecte definite
- [ ] Dataset recenzii: 300–600 înregistrări curate în `data/processed/`
- [ ] Lista finală de 10 aspecte + dicționar cuvinte cheie
- [ ] 200 exemple adnotate manual (2 membri independent → Cohen's Kappa)
- [ ] Primele statistici de întârziere per linie disponibile

### Săptămâna 3 — Extracție aspecte funcțională
- [ ] Pipeline spaCy extracție aspecte rulează end-to-end
- [ ] Evaluare manuală pe 100 exemple → Precision extracție
- [ ] Notebook EDA: distribuție aspecte, distribuție sentimente inițiale
- [ ] `notebooks/01_eda.ipynb` și `notebooks/02_aspect_extraction.ipynb` completate

### Săptămâna 4 — Modele sentiment + metrici
- [ ] Baseline lexical implementat și evaluat (F1 per aspect)
- [ ] BERT-ro fine-tuned, comparație cu baseline
- [ ] Tabel metrici final în `reports/model_results.md`
- [ ] Corelare pilot sentiment ↔ întârzieri (primele rezultate)

### Săptămâna 5 — Dashboard + validare completă
- [ ] Dashboard Streamlit funcțional cu hartă și heatmap
- [ ] Analiză corelare completă per linie și interval orar
- [ ] Demo intern al echipei — teste utilizabilitate
- [ ] Draft raport civic

### Săptămâna 6 — Finalizare
- [ ] README complet și documentație tehnică
- [ ] Raport civic final (`reports/civic_report.pdf`)
- [ ] Slide deck prezentare
- [ ] Code review final, cleanup, tag v1.0

---

## Convenții de lucru

**Branches:** `main` (stabil) · `dev` (integrare) · `feature/<nume>-<descriere>` (lucru individual)

**Commits:** mesaje în engleză, format `type: descriere scurtă`
- `feat: add gtfs poller script`
- `fix: handle missing diacritics in cleaner`
- `docs: update aspect lexicon documentation`

**Pull Requests:** cel puțin 1 review din echipă înainte de merge în `dev`

**Întâlniri:** săptămânal, înainte de fiecare milestone — sync de 30 min

---

## Structura notebook-urilor

| Notebook | Owner | Conținut |
|----------|-------|---------|
| `01_eda.ipynb` | toți | Explorare dataset, distribuții, exemple |
| `02_aspect_extraction.ipynb` | Talida | Testare spaCy, rafinare dicționar |
| `03_model_training.ipynb` | Andrei P. | Training BERT-ro, curbe loss |
| `04_evaluation.ipynb` | Andrei P. | Metrici, confusion matrix, comparație |
| `05_gtfs_analysis.ipynb` | Andrei A. | Analiza întârzierilor per linie |
| `06_correlation.ipynb` | Matei | Corelare sentiment ↔ GTFS |
