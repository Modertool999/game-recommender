import os
import json
import sqlite3
import requests
from pathlib import Path

# Adjust this if you want fewer/more than 20 000
TOP_N = 20000

POPULAR_URL = "https://api.steampowered.com/ISteamChartsService/GetMostPlayedGames/v1/"
DETAILS_URL = "https://store.steampowered.com/api/appdetails"
DB_PATH     = Path("data/steam_catalog.db")
CACHE_DIR   = Path("data/cache")

# Ensure folders exist
DB_PATH.parent.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# 1) Fetch top N most-played IDs
resp = requests.get(POPULAR_URL)
resp.raise_for_status()
ranks = resp.json().get("response", {}).get("ranks", [])
top_appids = [int(item["appid"]) for item in ranks[:TOP_N]]

# 2) Open (or create) SQLite DB
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS catalog (
    appid       INTEGER PRIMARY KEY,
    name        TEXT,
    description TEXT,
    genres      TEXT
)
""")
conn.commit()

# 3) Fetch details & insert
for appid in top_appids:
    cache_file = CACHE_DIR / f"app_{appid}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        r = requests.get(DETAILS_URL, params={"appids": appid})
        if not r.ok:
            # skip if no data
            continue
        entry = r.json().get(str(appid), {})
        data = entry.get("data", {}) or {}
        cache_file.write_text(json.dumps(data))

    name = data.get("name", "")
    desc = data.get("short_description", "")
    genres = ",".join(g.get("description", "") for g in data.get("genres", []))

    c.execute("""
      INSERT OR REPLACE INTO catalog (appid, name, description, genres)
      VALUES (?, ?, ?, ?)
    """, (appid, name, desc, genres))

conn.commit()
conn.close()

print(f"Stored top {TOP_N} games in {DB_PATH}")
