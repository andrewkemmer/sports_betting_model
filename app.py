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
# Streamlit Config
# --------------------------------------------------
st.set_page_config(page_title="Sports Betting EV + ML Model", layout="wide")
st.title("📈 Sports Betting EV Model with ML Integration")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("TheOddsAPI Settings")
api_key = st.sidebar.text_input("API Key", type="password")
sport_key = st.sidebar.selectbox(
    "Sport",
    ["baseball_mlb", "basketball_nba", "americanfootball_nfl", "icehockey_nhl"]
)
region = st.sidebar.selectbox("Region", ["us", "us2", "eu", "uk"])
market = st.sidebar.selectbox("Market", ["h2h", "spreads", "totals"])
btn_fetch = st.sidebar.button("Fetch Live Odds")

st.sidebar.header("Model Management")
btn_retrain = st.sidebar.button("Retrain Model with Latest Results")

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def fetch_live_odds(api_key, sport, region, market, odds_format="american"):
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        f"?apiKey={api_key}&regions={region}&markets={market}&oddsFormat={odds_format}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Error fetching odds: {r.status_code} - {r.text}")
        return None

    data = r.json()
    rows = []
    for game in data:
        gid = game["id"]
        home = game["home_team"]
        away = game["away_team"]

        # Always attach home_team and away_team to each row
        for book in game["bookmakers"]:
            book_name = book["title"]
            for mk in book["markets"]:
                outcomes = mk["outcomes"]
                for o in outcomes:
                    rows.append({
                        "game_id": gid,
                        "bookmaker": book_name,
                        "market": mk["key"],
                        "team": o["name"],
                        "opponent": away if o["name"] == home else home,
                        "line": o.get("point"),
                        "price": o["price"],
                        "home_team": home,
                        "away_team": away
                    })
    return pd.DataFrame(rows)

def fetch_scores_with_odds(api_key, sport="basketball_nba", days_back=3):
    all_rows = []
    for d in range(1, days_back+1):
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores/?apiKey={api_key}&daysFrom={d}"
        r = requests.get(url)
        if r.status_code != 200:
            continue
        data = r.json()
        for game in data:
            home = game["home_team"]
            away = game["away_team"]
            scores = game.get("scores", [])
            home_score = scores[0]["score"] if scores else None
            away_score = scores[1]["score"] if scores else None
            row = {
                "date": game["commence_time"],
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "spread_close": None,
                "total_close": None
            }
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market["key"] == "spreads":
                        for o in market["outcomes"]:
                            row["spread_close"] = o["point"]
                    elif market["key"] == "totals":
                        for o in market["outcomes"]:
                            row["total_close"] = o["point"]
            all_rows.append(row)
    return pd.DataFrame(all_rows)

def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def remove_vig(prob_a, prob_b):
    total = prob_a + prob_b
    return prob_a / total, prob_b / total

def ev_calc(prob, odds):
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / -odds
    return prob * payout - (1 - prob)

def retrain_and_log(df, sport="basketball_nba"):
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

def plot_calibration_curve(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    st.pyplot(fig)

# --------------------------------------------------
# Main Processing
# --------------------------------------------------
if btn_fetch:
    if not api_key:
        st.warning("Please enter your API key.")
    else:
        df = fetch_live_odds(api_key, sport_key, region, market)
        if df is not None:
            st.subheader("🔍 Raw Odds")
            st.dataframe(df)

            # Load model if available
            try:
                model = joblib.load(f"{sport_key}_model.pkl")
            except:
                model = None

            if model is not None:
                # Step 1: Build per-game features
                games = df.groupby("game_id").agg({
                    "line": "mean",
                    "home_team": "first",
                    "away_team": "first"
                }).reset_index()
                games.rename(columns={"line": "spread_close"}, inplace=True)
                games["total_close"] = games["spread_close"]  # placeholder until totals parsed separately

                # Step 2: Predict home win probability
                X_live = games[["spread_close", "total_close"]].fillna(0)
                games["model_prob_home_win"] = model.predict_proba(X_live)[:, 1]

                # Step 3: Merge back to odds table
                df = df.merge(
                    games[["game_id", "home_team", "away_team", "model_prob_home_win"]],
                    on="game_id",
                    how="left"
                )

                # Step 4: Assign probabilities to each team row safely
                if "home_team" in df.columns:
                    df["model_prob"] = np.where(
                        df["team"] == df["home_team"],
                        df["model_prob_home_win"],
                        1 - df["model_prob_home_win"]
                    )
                else:
                    st.warning("home_team column missing after merge — check API response structure.")

            # EV calculation with no-vig
            results = []
            grouped = df.groupby(["game_id", "bookmaker"])
            for (gid, book), g in grouped:
                if len(g) != 2:
                    continue
                t1, t2 = g.iloc[0], g.iloc[1]
                p1 = american_to_prob(t1["price"])
                p2 = american_to_prob(t2["price"])
                nv1, nv2 = remove_vig(p1, p2)

                results.append({
                    "team": t1["team"],
                    "odds": t1["price"],
                    "no_vig_prob": nv1,
                    "model_prob": t1["model_prob"],
                    "EV_model": ev_calc(t1["model_prob"], t1["price"])
                })

                results.append({
                    "team": t2["team"],
                    "odds": t2["price"],
                    "no_vig_prob": nv2,
                    "model_prob": t2["model_prob"],
