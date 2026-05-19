# Surse de date — CTP Pulse

## Sursa 1: tranzy.ai GTFS Real-time

**URL:** https://api.tranzy.ai/v1/opendata  
**Tip:** API REST + GTFS Realtime (Protocol Buffers)  
**Actualizare:** la fiecare 20–30 secunde  
**Autentificare:** API key (obținut din contul tranzy.ai)

### Date disponibile
- **Vehicle positions** — latitudine, longitudine, linie, vehicul ID, timestamp
- **Trip updates** — întârzieri față de orar, stații viitoare
- **Static GTFS** — rute, stații, orare planificate (ZIP descărcabil)

### Calcul întârziere
```
întârziere_sec = arrival_time_real - arrival_time_scheduled
```
Valori pozitive = autobuzul întârzie. Valori negative = autobuzul e înainte de orar.

### Format output (data/gtfs/)
```
gtfs_log_YYYY-MM-DD.csv
  trip_id, route_id, stop_id, scheduled_time, actual_time, delay_sec, lat, lon
```

---

## Sursa 2: Google Maps Places API

**Endpoint:** `places/details` + `places/search`  
**Autentificare:** API key Google Cloud Console  
**Cotă gratuită:** 200 USD/lună (~600 cereri places details)  
**Câmpuri relevante:** `reviews`, `rating`, `name`, `place_id`

### Locații țintă
- Pagina generală CTP Iași
- Stații principale: Piața Unirii, Gara CTP, Copou, Tătărași, Dacia etc.
- Se obțin place_id prin `places/search?query=statie+autobuz+iasi`

### Format output (data/raw/google_maps/)
```
reviews_google_YYYY-MM-DD.json
  [{place_id, place_name, author, rating, text, time, language}]
```

---

## Sursa 3: Facebook CTP Iași

**Pagină:** https://www.facebook.com/CTPublicIasi  
**Metodă:** facebook-scraper (scraping public, fără autentificare)  
**Date colectate:** comentarii la postări publice, rating pagină

### Limitări
- Facebook limitează scraping-ul — rulează cu delay între cereri
- Colectează maxim 100 comentarii per postare
- Doar postări și comentarii publice

### Format output (data/raw/facebook/)
```
comments_facebook_YYYY-MM-DD.json
  [{post_id, post_date, comment_id, comment_text, likes, date}]
```

---

## Sursa 4: App Store / Google Play (tranzy.app)

**App Store:** `app-store-scraper` (Python)  
**Google Play:** `google-play-scraper` (Python)  
**App ID tranzy:** `app.tranzy` (verifică în store)

### Format output (data/raw/app_reviews/)
```
reviews_appstore_YYYY-MM-DD.json
reviews_playstore_YYYY-MM-DD.json
  [{review_id, author, rating, title, text, date, version}]
```

---

## Schema unificată (data/processed/)

După preprocesare, toate sursele se unifică în:

```
reviews_unified.csv
  id, source, text_raw, text_clean, rating, date, language, line_mentioned
```

Câmpul `line_mentioned` e extras prin regex din text (ex: "linia 43", "autobuzul 8").
