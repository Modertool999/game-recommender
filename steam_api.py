import os
import requests
import pandas as pd

class SteamAPI:
    BASE  = "https://api.steampowered.com"
    STORE = "https://store.steampowered.com/api"

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("Missing Steam API key")
        self.key = api_key

    def get_owned_games(self, steamid: str) -> list[dict]:
        """Returns list of owned games (with appid, name, playtime_forever, etc.)."""
        url = f"{self.BASE}/IPlayerService/GetOwnedGames/v1/"
        params = {
            "key": self.key,
            "steamid": steamid,
            "include_appinfo": True,
            "include_played_free_games": True
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("response", {}).get("games", [])

    def get_recently_played_games(self, steamid: str, count: int = 50) -> list[dict]:
        """Returns list of games played in the last two weeks (appid, playtime_2weeks, etc.)."""
        url = f"{self.BASE}/IPlayerService/GetRecentlyPlayedGames/v1/"
        params = {
            "key": self.key,
            "steamid": steamid,
            "count": count
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("response", {}).get("games", [])

    def get_friends(self, steamid: str) -> list[str]:
        """Returns list of friend SteamIDs."""
        url = f"{self.BASE}/ISteamUser/GetFriendList/v1/"
        params = {
            "key": self.key,
            "steamid": steamid,
            "relationship": "friend"
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        friends = r.json().get("friendslist", {}).get("friends", [])
        return [f["steamid"] for f in friends]

    def get_app_details(self, appid: int) -> dict:
        """Fetches store metadata for a given app; skips on error."""
        url = f"{self.STORE}/appdetails"
        try:
            r = requests.get(url, params={"appids": appid}, timeout=5)
        except requests.RequestException:
            return {"appid": appid, "name": "", "description": "", "genres": ""}
        if not r.ok or not r.text.strip():
            return {"appid": appid, "name": "", "description": "", "genres": ""}
        try:
            payload = r.json()
        except ValueError:
            return {"appid": appid, "name": "", "description": "", "genres": ""}
        data = payload.get(str(appid), {}).get("data", {}) or {}
        return {
            "appid": appid,
            "name": data.get("name", ""),
            "description": data.get("short_description", ""),
            "genres": ",".join(g.get("description", "") for g in data.get("genres", []))
        }

def build_game_dataset(api_key: str, steamid: str) -> pd.DataFrame:
    """
    Returns a DataFrame of your owned games with columns:
    appid, name, description, genres.
    """
    steam = SteamAPI(api_key)
    owned = steam.get_owned_games(steamid)
    rows  = [steam.get_app_details(g["appid"]) for g in owned]
    return pd.DataFrame(rows)
