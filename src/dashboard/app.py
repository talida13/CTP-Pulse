"""
app.py
------
Dashboard Streamlit pentru vizualizarea rezultatelor CTP Pulse.

Owner: x
Rulare: streamlit run src/dashboard/app.py

Componente:
- Hartă Iași (folium) cu liniile CTP colorate după sentiment dominant
- Heatmap linie × aspect (plotly)
- Filtre interactive: linie, aspect, interval orar
- Tab separat: validare GTFS (sentiment vs. întârzieri reale)

TODO:
- [ ] Layout de bază Streamlit (sidebar filtre + main area)
- [ ] Componentă hartă folium cu linii colorate
- [ ] Heatmap plotly aspect × linie
- [ ] Integrare date din reports/
- [ ] Pagina "About" cu descrierea proiectului
"""
