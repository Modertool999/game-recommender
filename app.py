import os
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
from steam_api import build_game_dataset
from feature_builder import build_recent_playtime_features
from advanced_recommender import AdvancedRecommender

app = Flask(__name__, static_folder="static", template_folder="templates")

DB_PATH = "data/steam_catalog.db"
if not os.path.exists(DB_PATH):
    raise RuntimeError(f"Catalog DB not found at {DB_PATH}. Run catalog_db.py first.")
adv_rec = AdvancedRecommender(db_path=DB_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    steamid = request.args.get("steamid", type=str)
    if not steamid:
        return render_template("index.html", error="Please enter a Steam ID.")

    k = request.args.get("k", default=5, type=int)
    w1 = float(request.args.get("w1", default=0.6))
    w2 = float(request.args.get("w2", default=0.25))
    w3 = float(request.args.get("w3", default=0.15))
    total = w1 + w2 + w3
    if total > 0:
        w1, w2, w3 = w1 / total, w2 / total, w3 / total

    games_df = build_game_dataset(os.getenv("STEAM_API_KEY"), steamid)
    adv_rec.library = games_df["appid"].tolist()

    feats_df = build_recent_playtime_features(os.getenv("STEAM_API_KEY"), steamid)

    scaler = MinMaxScaler()
    if not feats_df.empty:
        feats_df[["my_playtime_norm", "friends_playtime_norm"]] = scaler.fit_transform(
            feats_df[["my_playtime", "friends_playtime"]]
        )
    else:
        feats_df["my_playtime_norm"] = 0
        feats_df["friends_playtime_norm"] = 0

    my_recent = feats_df[["appid", "my_playtime_norm"]].rename(columns={"my_playtime_norm": "playtime_2weeks"})
    friends_recent = feats_df[["appid", "friends_playtime_norm"]].rename(columns={"friends_playtime_norm": "playtime_2weeks"})
    
    recs_df = adv_rec.recommend(my_recent=my_recent, friends_recent=friends_recent, k=k, alpha=w1, beta=w2, gamma=w3)
    return render_template("index.html", recs=recs_df.to_dict(orient="records"))

if __name__ == "__main__":
    if not os.getenv("STEAM_API_KEY"):
        raise RuntimeError("Please set the STEAM_API_KEY environment variable.")
    app.run(debug=True)
