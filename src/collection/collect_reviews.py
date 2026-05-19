"""
collect_reviews.py
------------------
Colectează recenzii din Google Maps Places API pentru stațiile CTP Iași
și pagina generală CTP.

Owner: Afloroaiei Andrei-Gabriel
Output: data/raw/google_maps/reviews_google_YYYY-MM-DD.json

TODO:
- [ ] Configurare API key din .env
- [ ] Lista place_id stații principale
- [ ] Paginare recenzii (max 5 per cerere în API gratuit)
- [ ] Salvare JSON cu schema unificată
"""
