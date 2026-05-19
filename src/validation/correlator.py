"""
correlator.py
-------------
Corelează scorul de sentiment negativ per aspect ↔ întârzierea medie per linie CTP.

Owner: Pricop Matei-Ioan
Input: reviews_with_aspects.csv + gtfs_delays_summary.csv
Output: correlation_results.csv (Pearson/Spearman per aspect × linie)

Întrebarea de cercetare:
  "Oamenii se plâng de întârzieri — datele GTFS confirmă sau infirmă?"

TODO:
- [ ] Agregare sentiment negativ per linie per aspect
- [ ] Aggregare întârziere medie per linie (din GTFS)
- [ ] Calcul Pearson și Spearman
- [ ] Identificare outlieri (plângeri fără confirmare în date și invers)
"""
