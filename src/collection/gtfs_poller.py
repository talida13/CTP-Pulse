"""
gtfs_poller.py
--------------
Script de polling GTFS real-time de la tranzy.ai.
Rulează continuu și loghează pozițiile vehiculelor la fiecare GTFS_POLL_INTERVAL secunde.

Owner: x
Output: data/gtfs/gtfs_log_YYYY-MM-DD.csv

TODO:
- [ ] Conectare la tranzy.ai/opendata cu API key
- [ ] Parse Protocol Buffers cu gtfs-realtime-bindings
- [ ] Calcul întârziere: actual_time - scheduled_time
- [ ] Logging continuu în CSV cu timestamp
"""
