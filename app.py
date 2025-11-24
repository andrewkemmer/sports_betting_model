import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

# Sportsipy (team + boxscores)
from sportsipy.nba.teams import Teams
from sportsipy.nba.boxscore import Boxscores

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(page_title="Stat-Driven Model + EV (Sportsipy + TheOddsAPI)", layout="wide")
st.title("📈 Stat-Driven Model + EV using Sportsipy and Sportsbook Odds")

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("TheOddsAPI settings (for EV)")
api_key = st.sidebar.text_input("API Key", type="password")
sport_key = st.sidebar.selectbox(
    "Sport (odds source)",
    ["basketball_nba", "baseball_mlb", "americanfootball_nfl", "icehockey_nhl"],
    index=0
)
region = st.sidebar.selectbox("Region", ["us", "us2", "eu", "uk"])
market = st.sidebar.selectbox("Market", ["h2h"])  # EV focuses on moneyline h2h
btn_fetch = st.sidebar.button("Fetch live odds + EV")

st.sidebar.header("Model management (Sportsipy stats only)")
season_input = st.sidebar.text_input("Season (NBA, e.g., 2024)", value="2024")
days_back_train = st.sidebar.number_input("Days back for training (completed games)", min_value=7, max_value=120, value=45)
btn_retrain = st.sidebar.button("Retrain model from Sportsipy stats")

st.sidebar.header("Evaluation (NBA only)")
days_back_eval = st.sidebar.number_input("Days back for evaluation", min_value=7, max_value=120, value=30)
btn_evaluate = st.sidebar.button("Evaluate predictions on recent games")

# --------------------------------------------------
# Odds utilities (EV only)
# --------------------------------------------------
def american_to_prob(odds):
    if odds is None:
        return np.nan
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
    if pd.isna(prob) or odds is None:
        return np.nan
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
            book_name = book.get("title")
            for mk in book.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                outcomes = mk.get("outcomes", [])
                for o in outcomes:
                    rows.append({
                        "game_id": gid,
                        "bookmaker": book_name,
                        "team": o.get("name"),
                        "price": o.get("price"),
                        "home_team": home,
                        "away_team": away
                    })
    return pd.DataFrame(rows)

# --------------------------------------------------
# Sportsipy stats pipeline (NBA)
# --------------------------------------------------
def get_teams_stats_df(season: int) -> pd.DataFrame:
    teams = Teams(season)
    rows = []
    for t in teams:
        # Some attributes may be None; coerce to numeric safely
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
            "field_goals": n(getattr(t, "field_goals", np.nan)),
            "field_goal_attempts": n(getattr(t, "field_goal_attempts", np.nan)),
            "three_pointers": n(getattr(t, "three_pointers", np.nan)),
            "three_point_attempts": n(getattr(t, "three_point_attempts", np.nan)),
            "free_throws": n(getattr(t, "free_throws", np.nan)),
            "free_throw_attempts": n(getattr(t, "free_throw_attempts", np.nan)),
            "offensive_rebounds": n(getattr(t, "offensive_rebounds", np.nan)),
            "defensive_rebounds": n(getattr(t, "defensive_rebounds", np.nan)),
            "total_rebounds": n(getattr(t, "total_rebounds", np.nan)),
            "assists": n(getattr(t, "assists", np.nan)),
            "steals": n(getattr(t, "steals", np.nan)),
            "blocks": n(getattr(t, "blocks", np.nan)),
            "turnovers": n(getattr(t, "turnovers", np.nan)),
            "personal_fouls": n(getattr(t, "personal_fouls", np.nan)),
        })
    df = pd.DataFrame(rows)
    # Simple derived rates (robust against missing)
    df["fg_pct"] = (df["field_goals"] / df["field_goal_attempts"]).replace([np.inf, -np.inf], np.nan)
    df["three_pct"] = (df["three_pointers"] / df["three_point_attempts"]).replace([np.inf, -np.inf], np.nan)
    df["ft_pct"] = (df["free_throws"] / df["free_throw_attempts"]).replace([np.inf, -np.inf], np.nan)
    return df

def fetch_completed_games(days_back: int) -> pd.DataFrame:
    # Pull NBA boxscores from date range
    end = datetime.utcnow().date()
    start = end - timedelta(days=days_back)
    # Boxscores takes dates as YYYYMMDD strings
    def s(d): return d.strftime("%Y%m%d")
    box = Boxscores(s(start), s(end), nba=True)
    # box.games is a dict keyed by date with list of games; we flatten
    rows = []
    for date_str, games in box.games.items():
        for g in games:
            # g structure typically: {'home_name':..., 'away_name':..., 'home_score':..., 'away_score':..., ...}
            home = g.get("home_name")
            away = g.get("away_name")
            try:
                home_score = int(g.get("home_score")) if g.get("home_score") not in (None, "") else np.nan
                away_score = int(g.get("away_score")) if g.get("away_score") not in (None, "") else np.nan
            except Exception:
                home_score, away_score = np.nan, np.nan
            rows.append({
                "date": date_str,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score
            })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    return df

def build_matchup_features(games_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
    # Left join stats onto home/away teams and create differential features
    df = games_df.copy()
    df = df.merge(team_stats_df.add_prefix("home_"), left_on="home_team", right_on="home_team_name", how="left")
    df = df.merge(team_stats_df.add_prefix("away_"), left_on="away_team", right_on="away_team_name", how="left")

    # Feature diffs (home minus away)
    def diff(col):
        return f"{col}_diff"

    feature_cols_base = [
        "points_per_game", "opp_points_per_game",
        "offensive_rating", "defensive_rating", "pace",
        "fg_pct", "three_pct", "ft_pct",
        "total_rebounds", "assists", "steals", "blocks", "turnovers"
    ]

    for c in feature_cols_base:
        df[diff(c)] = df[f"home_{c}"] - df[f"away_{c}"]

    # Additional simple features
    df["win_pct_diff"] = (
        (df["home_wins"] / df["home_games_played"]) - (df["away_wins"] / df["away_games_played"])
    ).replace([np.inf, -np.inf], np.nan)

    # Target for training
    if {"home_score", "away_score"}.issubset(df.columns):
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Final feature set
    feature_cols = [diff(c) for c in feature_cols_base] + ["win_pct_diff"]
    X = df[feature_cols].fillna(0)
    return df, X, feature_cols

# --------------------------------------------------
# Model training + persistence (stats-only)
# --------------------------------------------------
def retrain_stats_model(season: int, days_back: int, sport_model_key="basketball_nba"):
    # Get team stats for season
    team_stats_df = get_teams_stats_df(season)

    # Get completed games for label
    games_df = fetch_completed_games(days_back)

    # Build features
    enriched_df, X, feature_cols = build_matchup_features(games_df, team_stats_df)
    y = enriched_df["home_win"].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    # Persist
    joblib.dump({"model": model, "feature_cols": feature_cols, "season": season, "team_stats_df": team_stats_df}, f"{sport_model_key}_stats_model.pkl")

    return model, X_test, y_test, y_prob, {"accuracy": acc, "brier_score": brier, "log_loss": ll}, team_stats_df

def plot_calibration_curve(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    st.pyplot(fig)

# --------------------------------------------------
# Prediction helpers (stats-only)
# --------------------------------------------------
def predict_matchups_from_odds_games(odds_games_df: pd.DataFrame, team_stats_df: pd.DataFrame, model_bundle_path: str):
    # odds_games_df: grouped per game_id, includes home_team/away_team
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Build features from teams present
    games_df = odds_games_df[["game_id", "home_team", "away_team"]].drop_duplicates()
    enriched_df, X, _ = build_matchup_features(games_df, team_stats_df)

    # Predict home win probability
    probs = model.predict_proba(X)[:, 1]
    enriched_df["model_prob_home_win"] = probs
    enriched_df["model_prob_away_win"] = 1 - probs
    enriched_df["game_id"] = games_df["game_id"].values
    return enriched_df[["game_id", "home_team", "away_team", "model_prob_home_win", "model_prob_away_win"]]

def compute_ev_table(odds_df: pd.DataFrame, model_preds_df: pd.DataFrame):
    # Prepare odds per game for home/away
    # We assume team names in odds_df match those in model_preds_df
    df = odds_df.copy()
    df["implied_prob"] = df["price"].apply(american_to_prob)

    # Pivot to get home/away implied probs per game by bookmaker (take best or average)
    # We'll aggregate by game and team with mean (you can change to max for best line)
    agg = (
        df.groupby(["game_id", "team"])
          .agg({"implied_prob": "mean", "price": "mean", "home_team": "first", "away_team": "first"})
          .reset_index()
    )

    # Merge model predictions
    merged = agg.merge(model_preds_df, on=["game_id", "home_team", "away_team"], how="left")

    # Assign which side is home vs away for model probabilities
    merged["model_prob_this_team"] = np.where(
        merged["team"] == merged["home_team"], merged["model_prob_home_win"], merged["model_prob_away_win"]
    )

    # Remove vig per game: need both sides' implied probs
    # Compute vig-adjusted implied for each side within game_id
    def adjust_group(g):
        # get home/away implied
        # If multiple team labels exist, we'll try match by home_team/away_team
        home_row = g[g["team"] == g["home_team"].iloc[0]].head(1)
        away_row = g[g["team"] == g["away_team"].iloc[0]].head(1)
        if home_row.empty or away_row.empty:
            g["implied_prob_novig"] = np.nan
            return g
        pa, pb = home_row["implied_prob"].values[0], away_row["implied_prob"].values[0]
        adj_a, adj_b = remove_vig(pa, pb)
        # assign back
        g.loc[home_row.index, "implied_prob_novig"] = adj_a
        g.loc[away_row.index, "implied_prob_novig"] = adj_b
        return g

    merged = merged.groupby("game_id", group_keys=False).apply(adjust_group)

    # EV using model prob vs price (American odds payout)
    merged["ev"] = merged.apply(lambda r: ev_calc(r["model_prob_this_team"], r["price"]), axis=1)

    # Optional: EV using no-vig implied probabilities to sanity check edge
    # Not strictly EV, but can show model vs market probability delta
    merged["edge_prob"] = merged["model_prob_this_team"] - merged["implied_prob_novig"]

    # Final columns
    out = merged[[
        "game_id", "home_team", "away_team", "team", "price",
        "model_prob_this_team", "implied_prob", "implied_prob_novig", "edge_prob", "ev"
    ]].sort_values(["game_id", "team"]).reset_index(drop=True)

    return out

# --------------------------------------------------
# Evaluation (stats-only outcomes vs actuals)
# --------------------------------------------------
def evaluate_recent_games(season: int, days_back: int, model_bundle_path: str):
    # Use completed games as evaluation set
    team_stats_df = joblib.load(model_bundle_path)["team_stats_df"]
    games_df = fetch_completed_games(days_back)
    if games_df.empty:
        return None, None

    enriched_df, X, _ = build_matchup_features(games_df, team_stats_df)
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]

    y_true = enriched_df["home_win"].astype(int)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)

    # Build a table of predictions vs actuals
    out = enriched_df[["date", "home_team", "away_team"]].copy()
    out["home_score"] = enriched_df.get("home_score", np.nan)
    out["away_score"] = enriched_df.get("away_score", np.nan)
    out["model_prob_home_win"] = y_prob
    out["predicted_winner"] = np.where(y_prob >= 0.5, out["home_team"], out["away_team"])
    out["actual_winner"] = np.where(out["home_score"] > out["away_score"], out["home_team"], out["away_team"])

    metrics = {"accuracy": float(acc), "brier_score": float(brier), "log_loss": float(ll)}
    return metrics, out

# --------------------------------------------------
# UI: Retraining (stats-only)
# --------------------------------------------------
if btn_retrain:
    try:
        st.info("Retraining model on Sportsipy stats and recent completed games...")
        season_val = int(season_input)
        model, X_test, y_test, y_prob, metrics, team_stats_df = retrain_stats_model(season_val, days_back_train, sport_model_key="basketball_nba")
        st.success("Model retrained and saved: basketball_nba_stats_model.pkl")
        st.subheader("📊 Performance metrics (stats-only classification)")
        st.write(f"Accuracy: {metrics['accuracy']:.3f}")
        st.write(f"Brier Score: {metrics['brier_score']:.3f}")
        st.write(f"Log Loss: {metrics['log_loss']:.3f}")
        plot_calibration_curve(y_test, y_prob)
    except Exception as e:
        st.error(f"Retraining failed: {e}")

# --------------------------------------------------
# UI: Live odds + EV (using model probs)
# --------------------------------------------------
if btn_fetch:
    if not api_key:
        st.warning("Please enter your API key.")
    else:
        st.info("Fetching live odds...")
        df_odds = fetch_live_odds_h2h(api_key, sport_key, region=region)
        if df_odds is None or df_odds.empty:
            st.warning("No live odds returned.")
        else:
            st.subheader("🔍 Raw odds (moneyline)")
            st.dataframe(df_odds)

            # Try to load stats-only model
            try:
                bundle = joblib.load("basketball_nba_stats_model.pkl")
                team_stats_df = bundle["team_stats_df"]
                st.success("Loaded stats-only model.")
            except Exception:
                team_stats_df = None
                st.info("No trained stats-only model found. Retrain to enable EV calculations with model probabilities.")

            if team_stats_df is not None:
                # Group to per-game
                games = df_odds.groupby("game_id").agg({"home_team": "first", "away_team": "first"}).reset_index()
                preds = predict_matchups_from_odds_games(games, team_stats_df, "basketball_nba_stats_model.pkl")
                ev_table = compute_ev_table(df_odds, preds)

                st.subheader("🧮 EV table (model probabilities vs sportsbook odds)")
                st.dataframe(ev_table)

                # Simple highlights: top +EV sides
                top_ev = ev_table.sort_values("ev", ascending=False).head(20)
                st.subheader("⭐ Top +EV opportunities (by EV)")
                st.dataframe(top_ev[["home_team", "away_team", "team", "price", "model_prob_this_team", "implied_prob_novig", "edge_prob", "ev"]])

# --------------------------------------------------
# UI: Evaluation on recent completed games (stats-only)
# --------------------------------------------------
if btn_evaluate:
    try:
        st.info("Evaluating predictions on recent completed games (NBA)...")
        metrics, eval_df = evaluate_recent_games(int(season_input), days_back_eval, "basketball_nba_stats_model.pkl")
        if metrics is None:
            st.warning("No completed games found for the selected window.")
        else:
            st.subheader("📊 Evaluation metrics (stats-only)")
            st.write(f"Accuracy: {metrics['accuracy']:.3f}")
            st.write(f"Brier Score: {metrics['brier_score']:.3f}")
            st.write(f"Log Loss: {metrics['log_loss']:.3f}")

            st.subheader("🏁 Predicted vs actual winners")
            st.dataframe(eval_df)
    except Exception as e:
        st.error(f"Evaluation failed: {e}")


