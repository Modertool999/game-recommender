import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

from steam_api import build_game_dataset, SteamAPIError
from feature_builder import build_recent_playtime_features
from advanced_recommender import AdvancedRecommender

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# 1) Initialize your global catalog-based recommender
DB_PATH = "data/steam_catalog.db"
if not os.path.exists(DB_PATH):
    raise RuntimeError(f"Catalog DB not found at {DB_PATH}. Run catalog_db.py first.")
adv_rec = AdvancedRecommender(db_path=DB_PATH)

class UserFacingError(RuntimeError):
    """Raised when we should surface a friendly error to the client."""

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

    try:
        games_df = build_game_dataset(api_key, steamid)
    except SteamAPIError as exc:
        raise UserFacingError(str(exc)) from exc
    if "appid" in games_df.columns:
        adv_rec.library = games_df["appid"].dropna().tolist()
    else:
        adv_rec.library = []

    try:
        feats_df, feature_stats = build_recent_playtime_features(api_key, steamid)
    except SteamAPIError as exc:
        raise UserFacingError(str(exc)) from exc
    feature_stats = dict(feature_stats or {})

    if feats_df.empty:
        feats_df = pd.DataFrame(columns=["appid", "my_playtime", "friends_playtime"])
        feats_df["my_playtime_norm"] = 0.0
        feats_df["friends_playtime_norm"] = 0.0
    else:
        def _normalize(col: str, out_col: str):
            series = feats_df[col].astype(float)
            max_val = series.max()
            feats_df[out_col] = series / max_val if max_val > 0 else 0.0

        _normalize("my_playtime", "my_playtime_norm")
        _normalize("friends_playtime", "friends_playtime_norm")

    feature_stats["feats_rows"] = int(len(feats_df))

    my_recent = feats_df[["appid", "my_playtime_norm"]].rename(
        columns={"my_playtime_norm": "playtime_2weeks"}
    )
    friends_recent = feats_df[["appid", "friends_playtime_norm"]].rename(
        columns={"friends_playtime_norm": "playtime_2weeks"}
    )

    used_library_fallback = False
    if my_recent["playtime_2weeks"].sum() == 0:
        if "appid" in games_df.columns:
            owned_base = games_df.drop_duplicates(subset=["appid"]).copy()
        else:
            owned_base = pd.DataFrame(columns=["appid", "playtime_forever"])
        if "playtime_forever" not in owned_base.columns:
            owned_base["playtime_forever"] = 0.0
        owned_base["playtime_forever"] = owned_base["playtime_forever"].fillna(0).astype(float)
        owned = owned_base[owned_base["playtime_forever"] > 0]
        if owned.empty and not owned_base.empty:
            owned = owned_base.copy()
            owned["playtime_forever"] = 1.0
        if owned.empty:
            my_recent = pd.DataFrame(columns=["appid", "playtime_2weeks"])
        else:
            max_forever = owned["playtime_forever"].max()
            owned["playtime_2weeks"] = (
                owned["playtime_forever"] / max_forever if max_forever > 0 else 0.0
            )
            my_recent = owned[["appid", "playtime_2weeks"]]
        used_library_fallback = True

    recs_df, signal_meta = adv_rec.recommend(
        my_recent=my_recent,
        friends_recent=friends_recent,
        k=k,
        alpha=w1,
        beta=w2,
        gamma=w3
    )
    meta = {
        "features": {
            **feature_stats,
            "used_library_fallback": used_library_fallback,
            "my_recent_rows": int(len(my_recent)),
            "friends_recent_rows": int(len(friends_recent)),
        },
        **signal_meta,
    }

    return recs_df.to_dict(orient="records"), meta

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
        recs, meta = _compute_recommendations(steamid, k, w1, w2, w3)
    except UserFacingError as exc:
        return render_template("index.html", error=str(exc))
    except Exception as exc:  # surface API errors in the template
        return render_template("index.html", error=str(exc))

    return render_template("index.html", recs=recs, rec_meta=meta)

@app.get("/api/recommend")
def api_recommend():
    """
    Query params:
      steam_id (str, required)
      k (int, optional, default 10)
      w_content (float, default 0.6)
      w_playtime (float, default 0.4)
      w_social (float, default 0.0)
    Returns: [{"title": str, "score": float}, ...]
    """
    steam_id = request.args.get("steam_id", "").strip()
    if not steam_id:
        return jsonify({"error": "steam_id required"}), 400

    try:
        k = int(request.args.get("k", 10))
        w_content = float(request.args.get("w_content", 0.6))
        w_playtime = float(request.args.get("w_playtime", 0.4))
        w_social = float(request.args.get("w_social", 0.0))
    except ValueError:
        return jsonify({"error": "Invalid numeric parameter."}), 400

    try:
        recs, meta = _compute_recommendations(steam_id, k, w_content, w_playtime, w_social)
    except UserFacingError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "results": [
            {"title": item.get("name"), "score": float(item["score"])}
            for item in recs
        ],
        "meta": meta
    })

if __name__ == "__main__":
    if not os.getenv("STEAM_API_KEY"):
        raise RuntimeError("Please set the STEAM_API_KEY environment variable.")
    app.run(debug=True)
