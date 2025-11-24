
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="Sports Betting Model", layout="wide")
st.title("📈 Sports Betting Model — Predictions & Accuracy")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("TheOddsAPI Settings")
api_key = st.sidebar.text_input("API Key", type="password")
sport_key = st.sidebar.selectbox(
    "Sport",
    ["basketball_nba", "baseball_mlb", "americanfootball_nfl", "icehockey_nhl"]
)
region = st.sidebar.selectbox("Region", ["us", "us2", "eu", "uk"])
btn_fetch = st.sidebar.button("Fetch live odds")

st.sidebar.header("Model management")
btn_retrain = st.sidebar.button("Retrain model (last 30 days)")

# -------------------------
# Helpers: odds
# -------------------------
def fetch_live_odds(api_key, sport, region, odds_format="american"):
    markets = "h2h,spreads,totals"
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        f"?apiKey={api_key}&regions={region}&markets={markets}&oddsFormat={odds_format}"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        st.error(f"Error fetching odds: {r.status_code} - {r.text}")
        return pd.DataFrame()
    data = r.json()
    rows = []
    for game in data:
        gid = game.get("id")
        home = game.get("home_team")
        away = game.get("away_team")
        for book in game.get("bookmakers", []):
            for mk in book.get("markets", []):
                for o in mk.get("outcomes", []):
                    rows.append({
                        "game_id": gid,
                        "market": mk.get("key"),
                        "team": o.get("name"),
                        "line": o.get("point"),
                        "home_team": home,
                        "away_team": away
                    })
    return pd.DataFrame(rows)

def extract_game_lines(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        return pd.DataFrame(columns=["game_id", "home_team", "away_team", "spread_close", "total_close"])
    spreads = (odds_df[odds_df["market"] == "spreads"]
               .groupby("game_id", as_index=False)["line"].mean()
               .rename(columns={"line": "spread_close"}))
    totals = (odds_df[odds_df["market"] == "totals"]
              .groupby("game_id", as_index=False)["line"].mean()
              .rename(columns={"line": "total_close"}))
    teams = (odds_df.groupby("game_id", as_index=False)
             .agg(home_team=("home_team", "first"), away_team=("away_team", "first")))
    games = teams.merge(spreads, on="game_id", how="left").merge(totals, on="game_id", how="left")
    return games

# -------------------------
# Helpers: scores (completed games only)
# -------------------------
def fetch_scores_with_odds(api_key, sport="basketball_nba", days_back=30) -> pd.DataFrame:
    rows = []
    for d in range(1, days_back + 1):
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores/?apiKey={api_key}&daysFrom={d}"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            continue
        for game in r.json():
            # Only keep completed games with final scores
            if not game.get("completed"):
                continue

            home = game.get("home_team")
            away = game.get("away_team")
            date = game.get("commence_time")
            scores = game.get("scores", [])

            home_score, away_score = None, None
            if isinstance(scores, list):
                try:
                    s_home = next((s for s in scores if s.get("name") == home), None)
                    s_away = next((s for s in scores if s.get("name") == away), None)
                    home_score = int(s_home.get("score")) if s_home and s_home.get("score") else None
                    away_score = int(s_away.get("score")) if s_away and s_away.get("score") else None
                except Exception:
                    pass

            if home_score is None or away_score is None:
                continue

            rows.append({
                "game_id": game.get("id"),
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "spread_close": None,
                "total_close": None
            })

    df = pd.DataFrame(rows)

    # Deduplicate
    if "game_id" in df.columns and df["game_id"].notna().any():
        df = df.drop_duplicates(subset=["game_id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["home_team", "away_team", "date"], keep="last")

    return df

# -------------------------
# Model training
# -------------------------
def retrain_and_log(df: pd.DataFrame, sport="basketball_nba"):
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        raise ValueError("No completed games found to retrain.")

    for col in ["spread_close", "total_close"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[["spread_close", "total_close"]]
    y = (df["home_score"] > df["away_score"]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob))
    }

    joblib.dump(model, f"{sport}_model.pkl")
    return model, X_test, y_test, y_prob, metrics

# -------------------------
# Predictions
# -------------------------
def predict_scores_from_lines(df: pd.DataFrame, model):
    df = df.copy()
    if "spread_close" not in df.columns:
        df["spread_close"] = np.nan
    if "total_close" not in df.columns:
        df["total_close"] = np.nan

    df["spread_close"] = pd.to_numeric(df["spread_close"], errors="coerce")
    df["total_close"] = pd.to_numeric(df["total_close"], errors="coerce")

    mask_spread = df["spread_close"].notna()
    df["predicted_margin"] = np.nan
    df["model_prob_home_win"] = np.nan
    df["model_prob_away_win"] = np.nan

    if mask_spread.any():
        X = df.loc[mask_spread, ["spread_close", "total_close"]].fillna(0)
        probs = model.predict_proba(X)[:, 1]
        df.loc[mask_spread, "model_prob_home_win"] = probs
        df.loc[mask_spread, "model_prob_away_win"] = 1 - probs
        df.loc[mask_spread, "predicted_margin"] = df.loc[mask_spread, "spread_close"] * (0.5 + probs)

    mask_total = mask_spread & df["total_close"].notna()
    df["predicted_total"] = np.nan
    df["predicted_home_score"] = np.nan
    df["predicted_away_score"] = np.nan

    if mask_total.any():
        df.loc[mask_total, "predicted_total"] = df.loc[mask_total, "total_close"]
        df.loc[mask_total, "predicted_home_score"] = (
            df.loc[mask_total, "predicted_total"] + df.loc[mask_total, "predicted_margin"]
        ) / 2
        df.loc[mask_total, "predicted_away_score"] = (
            df
