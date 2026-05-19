"""
cleaner.py
----------
Curățare și normalizare text românesc pentru recenzii CTP.

Owner: x
Input: text brut din orice sursă
Output: text curat, normalizat, gata de tokenizare

TODO:
- [ ] Lowercase + strip whitespace
- [ ] Normalizare diacritice (ș/ț cu virgulă vs sedilă)
- [ ] Eliminare emoji sau mapare la token semantic (😡 → NEG_EMOJI)
- [ ] Eliminare URL-uri, mențiuni, hashtag-uri
- [ ] Deduplicare recenzii identice
- [ ] Filtrare recenzii prea scurte (< 5 cuvinte)
"""
