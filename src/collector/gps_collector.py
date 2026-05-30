"""
gps_collector.py
----------------
Polling Tranzy API la fiecare 30s și salvează pozițiile vehiculelor CTP în fișiere CSV zilnice.
Owner: x
Input:  Tranzy API (TRANZY_API_KEY din .env)
Output: data/gps/gps_YYYY-MM-DD.csv (câte un fișier per zi)
TODO:
- [ ] Autentificare cu API key din .env
- [ ] Fetch /vehicles endpoint și parsare răspuns JSON
- [ ] Salvare snapshot cu coloane: timestamp, vehicle_id, route_id, latitude, longitude, speed
- [ ] Loop continuu cu sleep(30), creare fișier nou la schimbarea zilei
- [ ] Logging erori de rețea fără a opri colectarea

"""