import requests
import pandas as pd
from requests import exceptions as req_exc


class SteamAPIError(RuntimeError):
    """Raised when Steam Web API calls fail in a user-visible way."""

class SteamAPI:
    BASE  = "https://api.steampowered.com"
    STORE = "https://store.steampowered.com/api"
    DEFAULT_TIMEOUT = (5, 30)  # (connect, read) seconds

    def __init__(self, api_key: str, timeout=None):
        if not api_key:
            raise RuntimeError("Missing Steam API key")
        self.key = api_key
        self.timeout = timeout or self.DEFAULT_TIMEOUT

    def _request(self, url: str, *, timeout=None, **kwargs):
        effective_timeout = timeout or self.timeout
        try:
            resp = requests.get(url, timeout=effective_timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except req_exc.Timeout as exc:
            raise SteamAPIError(
                "Steam API timed out before responding. Valve’s servers may be busy; please retry in a moment."
            ) from exc
        except req_exc.HTTPError as exc:
            status = exc.response.status_code if exc.response else "unknown"
            if status in (401, 403):
                raise SteamAPIError(
                    "Steam rejected the request. Make sure your Steam profile and friends list are public."
                ) from exc
            raise SteamAPIError(
                f"Steam API returned HTTP {status}. Please try again shortly."
            ) from exc
        except req_exc.RequestException as exc:
            raise SteamAPIError(
                "Network error while contacting Steam. Please check your connection and try again."
            ) from exc

    def get_owned_games(self, steamid: str) -> list[dict]:
        """Returns list of owned games (with appid, name, playtime_forever, etc.)."""
        url = f"{self.BASE}/IPlayerService/GetOwnedGames/v1/"
        params = {
            "key": self.key,
            "steamid": steamid,
            "include_appinfo": True,
            "include_played_free_games": True
        }
        r = self._request(url, params=params)
        return r.json().get("response", {}).get("games", [])

    def get_recently_played_games(self, steamid: str, count: int = 50) -> list[dict]:
        """Returns list of games played in the last two weeks (appid, playtime_2weeks, etc.)."""
        url = f"{self.BASE}/IPlayerService/GetRecentlyPlayedGames/v1/"
        params = {
            "key": self.key,
            "steamid": steamid,
            "count": count
        }
        r = self._request(url, params=params)
        return r.json().get("response", {}).get("games", [])

    def get_friends(self, steamid: str) -> list[str]:
        """Returns list of friend SteamIDs."""
        url = f"{self.BASE}/ISteamUser/GetFriendList/v1/"
        params = {
            "key": self.key,
            "steamid": steamid,
            "relationship": "friend"
        }
        r = self._request(url, params=params)
        friends = r.json().get("friendslist", {}).get("friends", [])
        return [f["steamid"] for f in friends]

    def get_app_details(self, appid: int) -> dict:
        """Fetches store metadata for a given app; skips on error."""
        url = f"{self.STORE}/appdetails"
        try:
            r = self._request(url, params={"appids": appid}, timeout=(3, 10))
        except SteamAPIError:
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
    Returns a DataFrame of owned games enriched with store metadata.
    Columns include at least:
      - appid
      - name (from owned-games API)
      - playtime_forever (lifetime minutes, from the owned-games API)
      - playtime_recent (two week minutes when available)
    """
    steam = SteamAPI(api_key)
    owned = steam.get_owned_games(steamid) or []
    rows = []
    for entry in owned:
        appid = entry.get("appid")
        if appid is None:
            continue
        rows.append({
            "appid": appid,
            "name": entry.get("name", ""),
            "playtime_forever": entry.get("playtime_forever", 0),
            "playtime_recent": entry.get("playtime_2weeks", 0),
        })

    return pd.DataFrame(rows)
