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

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(page_title="Sports Betting EV + ML Model", layout="wide")
st.title("📈 Sports Betting EV Model with ML Integration")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("TheOddsAPI Settings")
api_key = st.sidebar.text_input("API Key", type="password")
sport_key = st.sidebar.selectbox(
    "Sport",
    ["baseball_mlb", "basketball_nba", "americanfootball_nfl", "icehockey_nhl"]
)
region = st.sidebar.selectbox("Region", ["us", "us2", "eu", "uk"])
btn_fetch = st.sidebar.button("Fetch live odds")

st.sidebar.header("Model management")
btn_retrain = st.sidebar.button("Retrain Model (last 30 days)")

# --------------------------------------------------
# Helpers: live odds and lines
# --------------------------------------------------
def fetch_live_odds(api_key, sport, region, odds_format="american"):
    markets = "h2h,spreads,totals"
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        f"?apiKey={api_key}&regions={region}&markets={markets}&oddsFormat={odds_format}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Error fetching odds: {r.status_code} - {r.text}")
        return None
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

def extract_game_lines(df):
    spreads = df[df["market"] == "spreads"].groupby("game_id")["line"].mean().rename("spread_close")
    totals = df[df["market"] == "totals"].groupby("game_id")["line"].mean().rename("total_close")
    games = df.groupby("game_id").agg({"home_team": "first", "away_team": "first"}).reset_index()
    games = games.merge(spreads.reset_index(), on="game_id", how="left")
    games = games.merge(totals.reset_index(), on="game_id", how="left")
    return games

# --------------------------------------------------
# Helpers: historical scores
# --------------------------------------------------
def fetch_scores_with_odds(api_key, sport="basketball_nba", days_back=30):
    all_rows = []
    for d in range(1, days_back + 1):
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores/?apiKey={api_key}&daysFrom={d}"
        r = requests.get(url)
        if r.status_code != 200:
            continue
        for game in r.json():
            home = game.get("home_team")
            away = game.get("away_team")
            scores = game.get("scores", [])
            home_score, away_score = None, None
            if isinstance(scores, list) and len(scores) >= 2:
                try:
                    s_home = next((s for s in scores if s.get("name") == home), scores[0])
                    s_away = next((s for s in scores if s.get("name") == away), scores[1])
                    home_score = int(s_home.get("score")) if s_home.get("score") else None
                    away_score = int(s_away.get("score")) if s_away.get("score") else None
                except Exception:
                    pass
            all_rows.append({
                "date": game.get("commence_time"),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "spread_close": None,
                "total_close": None
            })
    return pd.DataFrame(all_rows)

# --------------------------------------------------
# Model training and evaluation
# --------------------------------------------------
def retrain_and_log(df, sport="basketball_nba"):
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        raise ValueError("No completed games found to retrain.")
    for col in ["spread_close", "total_close"]:
        if col not in df.columns:
            df[col] = np.nan
    X = df[["spread_close", "total_close"]].fillna(0)
    y = (df["home_score"] > df["away_score"]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    joblib.dump(model, f"{sport}_model.pkl")
    return model, X_test, y_test, y_prob, {"accuracy": acc, "brier_score": brier, "log_loss": ll}

def predict_scores_from_lines(df, model):
    df = df.copy()
    for col in ["spread_close", "total_close"]:
        if col not in df.columns:
            df[col] = np.nan
    mask = df["spread_close"].notna() & df["total_close"].notna()
    if not mask.any():
        df["predicted_home_score"] = np.nan
        df["predicted_away_score"] = np.nan
        df["predicted_total"] = np.nan
        return df

    X = df.loc[mask, ["spread_close", "total_close"]].fillna(0)
    df.loc[mask, "model_prob_home_win"] = model.predict_proba(X)[:, 1]

    # Correct equations: home + away = total, home - away = margin
    df.loc[mask, "predicted_margin"] = df.loc[mask, "spread_close"] * (2 * df.loc[mask, "model_prob_home_win"] - 1)
    df.loc[mask, "predicted_total"] = df.loc[mask, "total_close"]

    df.loc[mask, "predicted_home_score"] = (df.loc[mask, "predicted_total"] + df.loc[mask, "predicted_margin"]) / 2
    df.loc[mask, "predicted_away_score"] = (df.loc[mask, "predicted_total"] - df.loc[mask, "predicted_margin"]) / 2

    df.loc[mask, "predicted_home_score"] = df.loc[mask, "predicted_home_score"].round(1)
    df.loc[mask, "predicted_away_score"] = df.loc[mask, "predicted_away_score"].round(1)
    df.loc[mask, "predicted_total"] = df.loc[mask, "predicted_total"].round(1)
    return df

def evaluate_predictions(df, model):
    df = predict_scores_from_lines(df, model)
    df["predicted_winner"] = np.where(df["predicted_home_score"] >= df["predicted_away_score"], df["home_team"], df["away_team"])
    df["actual_winner"] = np.where(df["home_score"] > df["away_score"], df["home_team"], df["away_team"])
    moneyline_acc = (df["predicted_winner"] == df["actual_winner"]).mean()
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["predicted_total_side"] = np.where(df["predicted_total"] > df["total_close"], "Over", "Under")
    df["actual_total_side"] = np.where(df["actual_total"] > df["total_close"], "Over", "Under")
    total_acc = (df["predicted_total_side"] == df["actual_total_side"]).mean()
    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["predicted_spread_cover"] = np.where(df["predicted_margin"] > df["spread_close"], "Home", "Away")
    df["actual_spread_cover"] = np.where(df["actual_margin"] > df["spread_close"], "Home", "Away")
    spread_acc = (df["predicted_spread_cover"] == df["actual_sp
