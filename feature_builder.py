import pandas as pd
from steam_api import SteamAPI

def build_recent_playtime_features(api_key: str, steamid: str, friends_limit: int = 50) -> pd.DataFrame:
    steam = SteamAPI(api_key)
    my_recent = steam.get_recently_played_games(steamid, count=100) or []
    if my_recent:
        df_my = pd.DataFrame(my_recent)[["appid", "playtime_2weeks"]]
        df_my = df_my.rename(columns={"playtime_2weeks": "my_playtime"})
    else:
        df_my = pd.DataFrame(columns=["appid", "my_playtime"])

    friends = steam.get_friends(steamid)[:friends_limit] or []
    rows = []
    for fid in friends:
        rec = steam.get_recently_played_games(fid, count=100) or []
        for g in rec:
            rows.append((g["appid"], g["playtime_2weeks"]))
    if rows:
        df_fr = pd.DataFrame(rows, columns=["appid", "playtime_2weeks"])
        df_fr = df_fr.groupby("appid").agg(friends_playtime=("playtime_2weeks", "sum"), friends_count=("playtime_2weeks", "count")).reset_index()
    else:
        df_fr = pd.DataFrame(columns=["appid", "friends_playtime", "friends_count"])
        
    feats = pd.merge(df_my, df_fr, on="appid", how="outer")

    for col in ["my_playtime", "friends_playtime", "friends_count"]:
        if col not in feats.columns:
            feats[col] = 0
    feats[["my_playtime", "friends_playtime", "friends_count"]] = (feats[["my_playtime", "friends_playtime", "friends_count"]].fillna(0).infer_objects(copy=False))

    return feats
