import os
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler

from steam_api import build_game_dataset
from feature_builder import build_recent_playtime_features
from advanced_recommender import AdvancedRecommender

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# 1) Initialize your global catalog-based recommender
DB_PATH = "data/steam_catalog.db"
if not os.path.exists(DB_PATH):
    raise RuntimeError(f"Catalog DB not found at {DB_PATH}. Run catalog_db.py first.")
adv_rec = AdvancedRecommender(db_path=DB_PATH)

def _compute_recommendations(steamid: str, k: int, w1: float, w2: float, w3: float):
    """Shared recommendation pipeline for HTML + JSON endpoints."""
    api_key = os.getenv("STEAM_API_KEY")
    if not api_key:
        raise RuntimeError("STEAM_API_KEY environment variable is required.")

    weights = [w1, w2, w3]
    total = sum(weights)
    if total > 0:
        w1, w2, w3 = [w / total for w in weights]

    k = max(1, k)

    games_df = build_game_dataset(api_key, steamid)
    adv_rec.library = games_df["appid"].tolist()

    feats_df = build_recent_playtime_features(api_key, steamid)

    scaler = MinMaxScaler()
    if not feats_df.empty:
        feats_df[["my_playtime_norm", "friends_playtime_norm"]] = scaler.fit_transform(
            feats_df[["my_playtime", "friends_playtime"]]
        )
    else:
        feats_df["my_playtime_norm"] = 0
        feats_df["friends_playtime_norm"] = 0

    my_recent = feats_df[["appid", "my_playtime_norm"]].rename(
        columns={"my_playtime_norm": "playtime_2weeks"}
    )
    friends_recent = feats_df[["appid", "friends_playtime_norm"]].rename(
        columns={"friends_playtime_norm": "playtime_2weeks"}
    )

    recs_df = adv_rec.recommend(
        my_recent=my_recent,
        friends_recent=friends_recent,
        k=k,
        alpha=w1,
        beta=w2,
        gamma=w3
    )

    return recs_df.to_dict(orient="records")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return {"ok": True}

@app.route("/recommend", methods=["GET"])
def recommend():
    steamid = request.args.get("steamid", type=str)
    if not steamid:
        return render_template("index.html", error="Please enter a Steam ID.")

    k = request.args.get("k", default=5, type=int)
    w1 = float(request.args.get("w1", default=0.6))
    w2 = float(request.args.get("w2", default=0.25))
    w3 = float(request.args.get("w3", default=0.15))

    try:
        recs = _compute_recommendations(steamid, k, w1, w2, w3)
    except Exception as exc:  # surface API errors in the template
        return render_template("index.html", error=str(exc))

    return render_template("index.html", recs=recs)

@app.get("/api/recommend")
def api_recommend():
    """
    Query params:
      steam_id (str, required)
      k (int, optional, default 10)
      w_content (float, default 0.6)
      w_playtime (float, default 0.4)
    Returns: [{"title": str, "score": float}, ...]
    """
    steam_id = request.args.get("steam_id", "").strip()
    if not steam_id:
        return jsonify({"error": "steam_id required"}), 400

    try:
        k = int(request.args.get("k", 10))
        w_content = float(request.args.get("w_content", 0.6))
        w_playtime = float(request.args.get("w_playtime", 0.4))
    except ValueError:
        return jsonify({"error": "Invalid numeric parameter."}), 400

    try:
        recs = _compute_recommendations(steam_id, k, w_content, w_playtime, 0.0)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify([
        {"title": item.get("name"), "score": float(item["score"])}
        for item in recs
    ])

if __name__ == "__main__":
    if not os.getenv("STEAM_API_KEY"):
        raise RuntimeError("Please set the STEAM_API_KEY environment variable.")
    app.run(debug=True)
