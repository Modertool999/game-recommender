import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedRecommender:
    def __init__(
        self,
        db_path: str = "data/steam_catalog.db",
        max_features: int = 50000
    ):
        # 1) Load catalog from SQLite
        conn = sqlite3.connect(db_path)
        self.catalog = pd.read_sql_query("SELECT * FROM catalog", conn)
        conn.close()

        # 2) Start with an empty library; to be set per-user
        self.library = []

        # 3) Build the combined text field
        self.catalog["text"] = (
            self.catalog["description"].fillna("") + " " +
            self.catalog["genres"].fillna("")
        )

        # 4) Fit a TF-IDF vectorizer over all catalog texts
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.catalog["text"])

    def recommend(
        self,
        my_recent: pd.DataFrame,
        friends_recent: pd.DataFrame,
        k: int = 10,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0
    ) -> pd.DataFrame:
        # 1) Filter out games already in library
        mask = ~self.catalog["appid"].isin(self.library)
        candidates = self.catalog[mask].reset_index(drop=True)
        cand_matrix = self.vectorizer.transform(candidates["text"])

        def profile_vec(recent_df: pd.DataFrame):
            idxs, weights = [], []
            for aid, wt in zip(recent_df["appid"], recent_df["playtime_2weeks"]):
                m = self.catalog["appid"] == aid
                if m.any():
                    idxs.append(int(self.catalog.index[m][0]))
                    weights.append(wt)
            if not idxs:
                return None
            mat = self.tfidf_matrix[idxs]
            wts = np.array(weights)[:, None]
            weighted = mat.multiply(wts)
            profile = weighted.sum(axis=0)
            arr = np.asarray(profile)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        vec_me = profile_vec(my_recent)
        vec_fr = profile_vec(friends_recent)

        sims_me = (
            cosine_similarity(vec_me, cand_matrix).flatten()
            if vec_me is not None else np.zeros(len(candidates))
        )
        sims_fr = (
            cosine_similarity(vec_fr, cand_matrix).flatten()
            if vec_fr is not None else np.zeros(len(candidates))
        )

        # direct friends’ playtime
        fr_map = friends_recent.set_index("appid")["playtime_2weeks"].to_dict()
        raw_fr = np.array([fr_map.get(aid, 0.0) for aid in candidates["appid"]], dtype=float)

        def _normalize_signal(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                return arr
            max_val = arr.max()
            min_val = arr.min()
            if np.isclose(max_val, min_val):
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)

        user_signal = _normalize_signal(sims_me)
        friends_play_signal = _normalize_signal(raw_fr)
        friends_sim_signal = _normalize_signal(sims_fr)

        candidates["score"] = (
            alpha * user_signal
            + beta * friends_play_signal
            + gamma * friends_sim_signal
        )
        return candidates.sort_values("score", ascending=False).head(k)[
            ["appid", "name", "score"]
        ]

if __name__ == "__main__":
    # smoke test
    from feature_builder import build_recent_playtime_features
    import os

    API_KEY = os.getenv("STEAM_API_KEY")
    STEAM_ID = os.getenv("STEAM_ID")
    feats = build_recent_playtime_features(API_KEY, STEAM_ID, friends_limit=30)

    my_recent = feats[["appid", "my_playtime"]].rename(
        columns={"my_playtime": "playtime_2weeks"}
    )
    fr_recent = feats[["appid", "friends_playtime"]].rename(
        columns={"friends_playtime": "playtime_2weeks"}
    )

    rec = AdvancedRecommender(db_path="data/steam_catalog.db")
    rec.library = []  # or load your library DataFrame’s appid list
    print(rec.recommend(my_recent, fr_recent, k=10, alpha=3.0, beta=2.0, gamma=1.0))
