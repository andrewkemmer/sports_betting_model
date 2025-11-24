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
        gid = game.get("id")
        home = game.get("home_team")
        away = game.get("away_team")
        for book in game.get("bookmakers", []):
            book_name = book.get("title")
            for mk in book.get("markets", []):
                for o in mk.get("outcomes", []):
                    rows.append({
                        "game_id": gid,
                        "bookmaker": book_name,
                        "market": mk.get("key"),
                        "team": o.get("name"),
                        "price": o.get("price"),
                        "line": o.get("point"),
                        "home_team": home,
                        "away_team": away
                    })
    return pd.DataFrame(rows)

def fetch_scores_with_odds(api_key, sport="basketball_nba", days_back=3):
    all_rows = []
    for d in range(1, days_back + 1):
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores/?apiKey={api_key}&daysFrom={d}"
        r = requests.get(url)
        if r.status_code != 200:
            continue
        data = r.json()
        for game in data:
            home = game.get("home_team")
            away = game.get("away_team")
            scores = game.get("scores", [])
            home_score = None
            away_score = None
            if isinstance(scores, list) and len(scores) >= 2:
                try:
                    s_home = next((s for s in scores if s.get("name") == home), scores[0])
                    s_away = next((s for s in scores if s.get("name") == away), scores[1])
                    home_score = int(s_home.get("score")) if s_home.get("score") else None
                    away_score = int(s_away.get("score")) if s_away.get("score") else None
                except Exception:
                    pass
            row = {
                "date": game.get("commence_time"),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "spread_close": None,
                "total_close": None
            }
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") == "spreads":
                        for o in market.get("outcomes", []):
                            if row["spread_close"] is None and o.get("point") is not None:
                                row["spread_close"] = o.get("point")
                    elif market.get("key") == "totals":
                        for o in market.get("outcomes", []):
                            if row["total_close"] is None and o.get("point") is not None:
                                row["total_close"] = o.get("point")
            all_rows.append(row)
    return pd.DataFrame(all_rows)

def retrain_and_log(df, sport="basketball_nba"):
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        raise ValueError("No completed games found to retrain.")
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

def evaluate_predictions(df, model):
    X = df[["spread_close", "total_close"]].fillna(0)
    df["model_prob_home_win"] = model.predict_proba(X)[:, 1]

    df["predicted_winner"] = np.where(df["model_prob_home_win"] >= 0.5, df["home_team"], df["away_team"])
    df["actual_winner"] = np.where(df["home_score"] > df["away_score"], df["home_team"], df["away_team"])
    moneyline_acc = (df["predicted_winner"] == df["actual_winner"]).mean()

    df["actual_total"] = df["home_score"] + df["away_score"]
    df["predicted_total_over"] = np.where(df["model_prob_home_win"] >= 0.5, "Over", "Under")
    df["actual_total_over"] = np.where(df["actual_total"] > df["total_close"], "Over", "Under")
    total_acc = (df["predicted_total_over"] == df["actual_total_over"]).mean()

    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["predicted_spread_cover"] = np.where(df["model_prob_home_win"] >= 0.5, "Home", "Away")
    df["actual_spread_cover"] = np.where(df["actual_margin"] > df["spread_close"], "Home", "Away")
    spread_acc = (df["predicted_spread_cover"] == df["actual_spread_cover"]).mean()

    return {
        "moneyline_accuracy": moneyline_acc,
        "total_accuracy": total_acc,
        "spread_accuracy": spread_acc,
        "df": df
    }

def save_accuracy_trends(results, sport="basketball_nba"):
    history_file = f"{sport}_accuracy_history.csv"
    new_row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "moneyline_accuracy": results["moneyline_accuracy"],
        "total_accuracy": results["total_accuracy"],
        "spread_accuracy": results["spread_accuracy"]
    }])
    try:
        history = pd.read_csv(history_file)
        history = pd.concat([history, new_row], ignore_index=True)
    except FileNotFoundError:
        history = new_row
    history.to_csv(history_file, index=False)
    return history

def plot_calibration_curve(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    st.pyplot(fig)

# --------------------------------------------------
# Retraining Dashboard
# --------------------------------------------------
if btn_retrain:
    if not api_key:
        st.warning("Please enter your API key.")
    else:
        st.info("Retraining model...")
        df_new = fetch_scores_with_odds(api_key, sport=sport_key, days_back=30)
        if df_new is None or df_new.empty:
            st.warning("No
