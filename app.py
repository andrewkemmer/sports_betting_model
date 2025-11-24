import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

# Sportsipy
from sportsipy.nba.teams import Teams
from sportsipy.nba.boxscore import Boxscores

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(page_title="Stat-Driven Model + EV", layout="wide")
st.title("📈 Stat-Driven Model + EV (Sportsipy + Sportsbook Odds)")

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("TheOddsAPI settings (for EV)")
api_key = st.sidebar.text_input("API Key", type="password")
sport_key = st.sidebar.selectbox("Sport (odds source)", ["basketball_nba"], index=0)
region = st.sidebar.selectbox("Region", ["us", "us2", "eu", "uk"])
btn_fetch = st.sidebar.button("Fetch live odds + EV")

st.sidebar.header("Model management (Sportsipy stats only)")
season_input = st.sidebar.text_input("Season (NBA, e.g., 2024)", value="2024")
days_back_train = st.sidebar.number_input("Days back for training", min_value=7, max_value=120, value=45)
btn_retrain = st.sidebar.button("Retrain model")

st.sidebar.header("Evaluation")
days_back_eval = st.sidebar.number_input("Days back for evaluation", min_value=7, max_value=120, value=30)
btn_evaluate = st.sidebar.button("Evaluate predictions")

# --------------------------------------------------
# Odds utilities
# --------------------------------------------------
def american_to_prob(odds):
    try:
        odds = float(odds)
    except Exception:
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def remove_vig(prob_a, prob_b):
    if pd.isna(prob_a) or pd.isna(prob_b):
        return np.nan, np.nan
    total = prob_a + prob_b
    if total <= 0:
        return np.nan, np.nan
    return prob_a / total, prob_b / total

def ev_calc(prob, odds):
    try:
        odds = float(odds)
    except Exception:
        return np.nan
    payout = (odds / 100) if odds > 0 else (100 / -odds)
    return prob * payout - (1 - prob)

def fetch_live_odds_h2h(api_key, sport, region="us", odds_format="american"):
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        f"?apiKey={api_key}&regions={region}&markets=h2h&oddsFormat={odds_format}"
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
                if mk.get("key") != "h2h":
                    continue
                for o in mk.get("outcomes", []):
                    rows.append({
                        "game_id": gid,
                        "bookmaker": book.get("title"),
                        "team": o.get("name"),
                        "price": o.get("price"),
                        "home_team": home,
                        "away_team": away
                    })
    return pd.DataFrame(rows)

# --------------------------------------------------
# Sportsipy stats pipeline
# --------------------------------------------------
def get_teams_stats_df(season: int) -> pd.DataFrame:
    teams = Teams(season)
    rows = []
    for t in teams:
        def n(x):
            try:
                return float(x) if x is not None else np.nan
            except Exception:
                return np.nan
        rows.append({
            "team_name": t.name,
            "games_played": n(t.games_played),
            "wins": n(t.wins),
            "losses": n(t.losses),
            "points_per_game": n(t.points_per_game),
            "opp_points_per_game": n(t.opp_points_per_game),
            "offensive_rating": n(getattr(t, "offensive_rating", np.nan)),
            "defensive_rating": n(getattr(t, "defensive_rating", np.nan)),
            "pace": n(getattr(t, "pace", np.nan)),
            "field_goal_pct": n(getattr(t, "field_goal_percentage", np.nan)),
            "three_point_pct": n(getattr(t, "three_point_field_goal_percentage", np.nan)),
            "free_throw_pct": n(getattr(t, "free_throw_percentage", np.nan)),
            "tot_reb_per_game": n(getattr(t, "total_rebounds_per_game", np.nan)),
            "assists_per_game": n(getattr(t, "assists_per_game", np.nan)),
            "steals_per_game": n(getattr(t, "steals_per_game", np.nan)),
            "blocks_per_game": n(getattr(t, "blocks_per_game", np.nan)),
            "turnovers_per_game": n(getattr(t, "turnovers_per_game", np.nan)),
        })
    return pd.DataFrame(rows)

def fetch_completed_games(days_back: int) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days_back)
    box = Boxscores(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), nba=True)
    rows = []
    for date_str, games in box.games.items():
        for g in games:
            home = g.get("home_name")
            away = g.get("away_name")
            try:
                home_score = int(g.get("home_score")) if g.get("home_score") else np.nan
                away_score = int(g.get("away_score")) if g.get("away_score") else np.nan
            except Exception:
                home_score, away_score = np.nan, np.nan
            rows.append({
                "date": date_str,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score
            })
    return pd.DataFrame(rows).dropna()

def build_matchup_features(games_df: pd.DataFrame, team_stats_df: pd.DataFrame):
    df = games_df.copy()
    df = df.merge(team_stats_df.add_prefix("home_"), left_on="home_team", right_on="home_team_name", how="left")
    df = df.merge(team_stats_df.add_prefix("away_"), left_on="away_team", right_on="away_team_name", how="left")

    feature_cols_base = [
        "points_per_game", "opp_points_per_game",
        "offensive_rating", "defensive_rating", "pace",
        "field_goal_pct", "three_point_pct", "free_throw_pct",
        "tot_reb_per_game", "assists_per_game", "steals_per_game",
        "blocks_per_game", "turnovers_per_game"
    ]

    for c in feature_cols_base:
        df[f"{c}_diff"] = df[f"home_{c}"] - df[f"away_{c}"]

    df["win_pct_diff"] = (
        (df["home_wins"] / df["home_games_played"]) - (df["away_wins"] / df["away_games_played"])
    ).replace([np.inf, -np.inf], np.nan)

    if {"home_score", "away_score"}.issubset(df.columns):
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    feature_cols = [f"{c}_diff" for c in feature_cols_base] + ["win_pct_diff"]
    X = df[feature_cols].fillna(0)
    return df, X, feature_cols

# --------------------------------------------------
# Model training
# --------------------------------------------------
def retrain_stats_model(season: int, days_back: int):
    team_stats_df = get_teams_stats_df(season)
    games_df = fetch_completed_games(days_back)
    enriched_df, X, feature_cols = build_matchup_features(games_df, team_stats_df)
    y = enriched_df["home_win"].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    # Save model bundle (includes stats for later predictions)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "season": season,
            "team_stats_df": team_stats_df
        },
        "basketball_nba_stats_model.pkl"
    )

    return model, X_test, y_test, y_prob, {"accuracy": acc, "brier_score": brier, "log_loss": ll}, team_stats_df

