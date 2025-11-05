import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

import pandas as pd

from steam_api import SteamAPI, SteamAPIError

DEFAULT_FRIEND_LIMIT = 30
MAX_FRIEND_WORKERS = 8


def build_recent_playtime_features(
    api_key: str,
    steamid: str,
    friends_limit: int = DEFAULT_FRIEND_LIMIT
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Returns a DataFrame with these columns:
      - appid
      - my_playtime       (sum of your playtime in the last 2 weeks)
      - friends_playtime  (sum of your friends' playtime in the last 2 weeks)
      - friends_count     (number of friends who played each game)
    Handles empty responses gracefully.
    """
    steam = SteamAPI(api_key)
    try:
        env_limit = int(os.getenv("STEAM_FRIEND_LIMIT", friends_limit))
        friends_limit = max(0, env_limit)
    except (TypeError, ValueError):
        friends_limit = max(0, friends_limit)

    # 1) Your recent plays
    my_recent = steam.get_recently_played_games(steamid, count=100) or []
    if my_recent:
        df_my = pd.DataFrame(my_recent)[["appid", "playtime_2weeks"]]
        df_my = df_my.rename(columns={"playtime_2weeks": "my_playtime"})
    else:
        # empty frame with correct columns
        df_my = pd.DataFrame(columns=["appid", "my_playtime"])

    # 2) Gather friends' recent plays
    friends = steam.get_friends(steamid)[:friends_limit] or []
    rows = []
    friends_requested = len(friends)
    friends_accessible = 0
    friends_with_recent = 0
    friends_failed = 0

    def _fetch_recent(fid: str):
        try:
            rec = steam.get_recently_played_games(fid, count=100) or []
            return "ok", rec
        except SteamAPIError:
            return "error", []

    if friends_requested:
        max_workers = min(MAX_FRIEND_WORKERS, friends_requested) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_fetch_recent, fid): fid for fid in friends}
            for fut in as_completed(future_map):
                status, rec = fut.result()
                if status != "ok":
                    friends_failed += 1
                    continue
                friends_accessible += 1
                if rec:
                    friends_with_recent += 1
                    for g in rec:
                        rows.append((g["appid"], g["playtime_2weeks"]))

    stats = {
        "friends_requested": friends_requested,
        "friends_accessible": friends_accessible,
        "friends_with_recent": friends_with_recent,
        "friends_failed": friends_failed,
        "friend_rows": len(rows),
        "my_recent_count": len(df_my),
        "friend_limit_effective": friends_limit,
    }
    if rows:
        df_fr = pd.DataFrame(rows, columns=["appid", "playtime_2weeks"])
        df_fr = df_fr.groupby("appid").agg(
            friends_playtime=("playtime_2weeks", "sum"),
            friends_count=("playtime_2weeks", "count")
        ).reset_index()
    else:
        df_fr = pd.DataFrame(columns=["appid", "friends_playtime", "friends_count"])

    # 3) Merge your plays and friends' plays
    feats = pd.merge(df_my, df_fr, on="appid", how="outer")

    # 4) Ensure columns exist and fill missing numeric values with zero
    for col in ["my_playtime", "friends_playtime", "friends_count"]:
        if col not in feats.columns:
            feats[col] = 0
    feats[["my_playtime", "friends_playtime", "friends_count"]] = (
        feats[["my_playtime", "friends_playtime", "friends_count"]]
        .fillna(0)
        .infer_objects(copy=False)
    )

    return feats, stats
