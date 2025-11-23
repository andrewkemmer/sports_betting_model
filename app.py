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
market = st.sidebar.selectbox("Market", ["h2h", "spreads", "totals"])
btn_fetch = st.sidebar.button("Fetch Live Odds")

st.sidebar.header("Model management")
btn_retrain = st.sidebar.button("Retrain model with latest results")

# --------------------------------------------------
# Helpers
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
    if total == 0:
        return np.nan, np.nan
    return prob_a / total, prob_b / total

def ev_calc(prob, odds):
    if pd.isna(prob) or odds is None:
        return np.nan
    try:
        odds = float(odds)
    except Exception:
        return np.nan
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / -odds
    return prob * payout - (1 - prob)

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
# Main processing
# --------------------------------------------------
if btn_fetch:
    if not api_key:
        st.warning("Please enter your API key.")
    else:
        df = fetch_live_odds(api_key, sport_key, region, market)
        if df is None or df.empty:
            st.warning("No live odds returned.")
        else:
            st.subheader("🔍 Raw Odds")
            st.dataframe(df)

            try:
                model = joblib.load(f"{sport_key}_model.pkl")
            except Exception:
                model = None
                st.info("No trained model found yet. Retrain to enable model probabilities.")

            if model is not None:
                # Build per-game features
                games = (
                    df.groupby("game_id")
                      .agg({"line": "mean", "home_team": "first", "away_team": "first"})
                      .reset_index()
                )
                games = games.rename(columns={"line": "spread_close"})
                games["total_close"] = games["spread_close"].fillna(0)
                X_live = games[["spread_close", "total_close"]].fillna(0)
                games["model_prob_home_win"] = model.predict_proba(X_live)[:, 1]

                # Merge back and ensure canonical team columns
                df = df.merge(
                    games[["game_id", "home_team", "away_team", "model_prob_home_win"]],
                    on="game_id",
                    how="left",
                    suffixes=("", "_games")
                )
                for col in ["home_team", "away_team"]:
                    alt = f"{col}_games"
                    if alt in df.columns:
                        df[col] = df[col].fillna(df[alt])
                        df.drop(columns=[alt], inplace=True)

                # Assign per-row team probability safely
                if {"home_team", "away_team"}.issubset(df.columns):
                    df["model_prob"] = np.where(
                        df["team"] == df["home_team"],
                        df["model_prob_home_win"],
                        1 - df["model_prob_home_win"]
                    )
                else:
                    df["model_prob"] = np.nan
                    st.warning("home_team/away_team missing after merge; model_prob set to NaN.")

            # EV calculation (pair first two outcomes per game-bookmaker)
            results = []
            grouped = df.groupby(["game_id", "bookmaker"])
            for (gid, book), g in grouped:
                if len(g) < 2:
                    continue
                g = g.head(2)
                t1, t2 = g.iloc[0], g.iloc[1]
                p1 = american_to_prob(t1["price"])
                p2 = american_to_prob(t2["price"])
                nv1, nv2 = remove_vig(p1, p2)
                results.append({
                    "game_id": gid,
                    "bookmaker": book,
                    "team": t1["team"],
                    "odds": t1["price"],
                    "no_vig_prob": nv1,
                    "model_prob": t1.get("model_prob", np.nan),
                    "EV_model": ev_calc(t1.get("model_prob", np.nan), t1["price"])
                })
                results.append({
                    "game_id": gid,
                    "bookmaker": book,
                    "team": t2["team"],
                    "odds": t2["price"],
                    "no_vig_prob": nv2,
                    "model_prob": t2.get("model_prob", np.nan),
                    "EV_model": ev_calc(t2.get("model_prob", np.nan), t2["price"])
                })

            out = pd.DataFrame(results)
            st.subheader("🎯 Model Probabilities vs. Book Odds")
            if out.empty:
                st.info("No paired outcomes found to compute EV. Try a different market or region.")
            else:
                st.dataframe(out)

# --------------------------------------------------
# Retraining
# --------------------------------------------------
if btn_retrain:
    if not api_key:
        st.warning("Please enter your API key.")
    else:
        st.info("Retraining model...")
        df_new = fetch_scores_with_odds(api_key, sport=sport_key, days_back=30)
        if df_new is None or df_new.empty:
            st.warning("No historical data retrieved.")
        else:
            try:
                model, X_test, y_test, y_prob, metrics = retrain_and_log(df_new, sport=sport_key)
                st.success("Model retrained!")
                st.write("📊 Performance Metrics:")
                st.write(f"Accuracy: {metrics['accuracy']:.3f}")
                st.write(f"Brier Score: {metrics['brier_score']:.3f}")
                st.write(f"Log Loss: {metrics['log_loss']:.3f}")
                plot_calibration_curve(y_test, y_prob)
            except Exception as e:
                st.error(f"Retraining failed: {e}")
