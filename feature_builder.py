import os
import pandas as pd
from steam_api import SteamAPI, SteamAPIError

DEFAULT_FRIEND_LIMIT = 12

def build_recent_playtime_features(
    api_key: str,
    steamid: str,
    friends_limit: int = DEFAULT_FRIEND_LIMIT
) -> pd.DataFrame:
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
    for fid in friends:
        try:
            rec = steam.get_recently_played_games(fid, count=100) or []
        except SteamAPIError:
            continue
        for g in rec:
            rows.append((g["appid"], g["playtime_2weeks"]))
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

    return feats
