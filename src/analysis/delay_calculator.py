"""
delay_calculator.py
-------------------
Pentru o linie și o dată dată, calculează întârzierea medie față de orarul teoretic.
Owner: x
Input:  data/timetable/timetable.json + data/gps/gps_YYYY-MM-DD.csv sau Tranzy API live
Output: dict cu route_id, date, avg_delay_minutes, max_delay_minutes, num_vehicles
TODO:
- [ ] Citire orar teoretic din timetable.json pentru linia cerută
- [ ] Citire pozițiile GPS pentru data cerută din fișierul CSV zilnic
- [ ] Matching vehicul → linie pe baza route_id din GPS
- [ ] Calcul diferență timestamp GPS față de ora teoretică cea mai apropiată din orar
- [ ] Agregare: medie, maxim, număr vehicule monitorizate
- [ ] Fallback dacă fișierul GPS pentru data respectivă nu există: returnează None
"""