# Steam Game Recommender

A Python/Flask application that delivers personalized Steam game recommendations by combining content-based filtering on game metadata with social playtime signals.

## Description

- Builds a one-time SQLite catalog if it doesn't already exist (`data/steam_catalog.db`) of the top 20 000 most-played Steam games (name, description, genres).  
- Aggregates each user’s and their friends’ recent playtime via the Steam Web API.  
- Computes TF-IDF vectors on game text and scores candidates with weighted signals:  
  1. Similarity to user’s recent plays  
  2. Friends’ aggregate playtime  
  3. Similarity to friends’ recent plays  

Weights are normalized and configurable.

## Structure

app.py : Flask routes & UI
steam_api.py:  Steam API wrapper
catalog_db.py: Builds/updates SQLite catalog
feature_builder.py: Builds recent-play features
advanced_recommender.py: TF-IDF model & scoring
templates/index.html: Form UI
static/styles.css: Basic styling

## Setup

1. Clone repo and enter directory.  
2. Create & activate a virtualenv:  
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
3. Install requirements:
```bash
pip install -r requirements.txt
```
4. Set your Steam API key by replacing YOUR_KEY with the key found at https://steamcommunity.com/dev/apikey:
```bash
export STEAM_API_KEY=YOUR_KEY
```
5. Build catalog (once):
```bash
python catalog_db.py
```
5. Run program
```
python app.py
```

- Visit http://127.0.0.1:5000/
- Enter your 64-bit Steam ID (lookup at https://steamid.io/lookup), desired count, and optional weights.
- Submit to view recommendations



