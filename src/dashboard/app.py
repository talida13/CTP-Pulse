"""
app.py
------
Aplicația Streamlit — interfața principală a proiectului.
Owner: x
Input:  data/ctp_pulse.db + model/ctp_absa_bert/ + data/timetable/timetable.json
Output: UI web accesibil la localhost:8501
TODO:
- [ ] Navigare între 3 pagini: Review nou, Dashboard, Căutare per linie
- [ ] Pagina Review nou: input text, buton Submit, afișare aspecte extrase cu culori
- [ ] Dacă aspect punctualitate detected: afișare rezultat GPS check automat
- [ ] Pagina Dashboard: heatmap plotly linie x aspect (culoare = % recenzii negative)
- [ ] Pagina Dashboard: grafic trend temporal sentiment per aspect pe luni
- [ ] Salvare review nou + rezultate ABSA în SQLite după submit
"""
