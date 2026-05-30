"""
aspect_extractor.py
-------------------
Extracție aspecte din text folosind spaCy ro_core_news_md
și lista de aspecte predefinite in aspects.md

Owner: x
Input: text curat (str)
Output: listă de (aspect_code, span_text) detectate în text

TODO:
- [ ] Încărcare model spaCy ro_core_news_md
- [ ] Extracție noun chunks
- [ ] Mapare chunk → aspect via dicționar
- [ ] Fallback: căutare directă cuvinte cheie
- [ ] Returnare context (fereastra de ±3 tokeni în jurul aspectului)
"""
