"""
correlate.py
------------
Pentru recenziile cu aspectul punctualitate, verifică dacă întârzierea e confirmată de GPS.
Owner: 
Input:  data/processed/adnotare_gold.csv + data/gps/ + data/timetable/timetable.json
Output: data/processed/correlation_results.csv
Întrebarea de cercetare:
  "Oamenii se plâng de întârzieri — datele GPS confirmă sau infirmă?"
TODO:
- [ ] Filtrare recenzii cu aspect punctualitate și sentiment negativ
- [ ] Extragere linie din textul recenziei cu regex (linia 8, tramvaiul 3 etc.)
- [ ] Apel delay_calculator.py pentru linia și data recenziei
- [ ] Marcare fiecare recenzie: confirmed / infirmed / insufficient_data
- [ ] Statistică finală: % recenzii de întârziere confirmate de GPS
"""
