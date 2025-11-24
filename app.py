
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
# Sidebar inputs
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
# Helper functions
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

def fetch_scores_with_odds(api_key, sport="basketball_nba", days_back=30):
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
    # Ensure required columns exist
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
    """
    Create predicted final scores using spread and total lines, adjusted by model home-win probability.
    Assumptions:
      - predicted_total ≈ total_close
      - predicted_margin ≈ spread_close adjusted by model confidence: margin_adj = spread_close * (2*p_home - 1)
    """
    df = df.copy()
    X = df[["spread_close", "total_close"]].fillna(0)
    df["model_prob_home_win"] = model.predict_proba(X)[:, 1]

    # Predicted margin (home - away) adjusted by confidence
    df["predicted_margin"] = df["spread_close"].fillna(0) * (2 * df["model_prob_home_win"] - 1)
    # Predicted total
    df["predicted_total"] = df["total_close"].fillna(df["home_score"] + df["away_score"] if "home_score" in df.columns else 0)

    # Predicted scores
    df["predicted_home_score"] = (df["predicted_total"] + df["predicted_margin"]) / 2
    df["predicted_away_score"] = df["predicted_total"] - df["predicted_home_score"]

    # Round for display
    df["predicted_home_score"] = df["predicted_home_score"].round(1)
    df["predicted_away_score"] = df["predicted_away_score"].round(1)
    return df

def evaluate_predictions(df, model):
    """
    Return accuracy metrics and a detailed dataframe with predictions vs actuals:
      - Moneyline winner accuracy
      - Total (Over/Under) accuracy vs total_close
      - Spread (ATS) accuracy vs spread_close
    """
    df = predict_scores_from_lines(df, model)

    # Moneyline winner
    df["predicted_winner"] = np.where(df["predicted_home_score"] >= df["predicted_away_score"], df["home_team"], df["away_team"])
    df["actual_winner"] = np.where(df["home_score"] > df["away_score"], df["home_team"], df["away_team"])
    moneyline_acc = (df["predicted_winner"] == df["actual_winner"]).mean()

    # Total Over/Under
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["predicted_total_side"] = np.where(df["predicted_total"] > df["total_close"], "Over", "Under")
    df["actual_total_side"] = np.where(df["actual_total"] > df["total_close"], "Over", "Under")
    total_acc = (df["predicted_total_side"] == df["actual_total_side"]).mean()

    # Spread (ATS)
    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["predicted_spread_cover"] = np.where(df["predicted_margin"] > df["spread_close"], "Home", "Away")
    df["actual_spread_cover"] = np.where(df["actual_margin"] > df["spread_close"], "Home", "Away")
    spread_acc = (df["predicted_spread_cover"] == df["actual_spread_cover"]).mean()

    return {
        "moneyline_accuracy": float(moneyline_acc) if not pd.isna(moneyline_acc) else 0.0,
        "total_accuracy": float(total_acc) if not pd.isna(total_acc) else 0.0,
        "spread_accuracy": float(spread_acc) if not pd.isna(spread_acc) else 0.0,
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
# Live odds view (optional EV)
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

            # Try load model
            try:
                model = joblib.load(f"{sport_key}_model.pkl")
            except Exception:
                model = None
                st.info("No trained model found yet. Retrain to enable predictions.")

            if model is not None:
                # Per-game features from current odds to show predicted scores on live games
                games = (
                    df.groupby("game_id")
                      .agg({"line": "mean", "home_team": "first", "away_team": "first"})
                      .reset_index()
                ).rename(columns={"line": "spread_close"})
                # If totals market is present, try to pull representative total_close per game
                totals = df[df["market"] == "totals"].groupby("game_id")["line"].mean().rename("total_close")
                games = games.merge(totals.reset_index(), on="game_id", how="left")
                games["total_close"] = games["total_close"].fillna(games["spread_close"].abs() * 4).clip(lower=0)  # simple fallback

                # Predicted scores using current model + lines
                pred_live = predict_scores_from_lines(games.assign(home_score=np.nan, away_score=np.nan), model)
                st.subheader("🧮 Live predicted scores (model + market lines)")
                st.dataframe(pred_live[["home_team", "away_team", "predicted_home_score", "predicted_away_score", "predicted_total", "predicted_margin"]])

# --------------------------------------------------
# Retraining + accuracy dashboard
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
                st.subheader("📊 Performance metrics (classification)")
                st.write(f"Accuracy: {metrics['accuracy']:.3f}")
                st.write(f"Brier Score: {metrics['brier_score']:.3f}")
                st.write(f"Log Loss: {metrics['log_loss']:.3f}")
                plot_calibration_curve(y_test, y_prob)

                # Evaluate predictions vs actuals (moneyline, total, spread) + predicted scores
                eval_results = evaluate_predictions(df_new, model)

                st.subheader("🏁 Predicted vs actual outcomes (last 30 days)")
                st.metric("Moneyline winner accuracy", f"{eval_results['moneyline_accuracy']:.2%}")
                st.metric("Total score accuracy (O/U vs total_close)", f"{eval_results['total_accuracy']:.2%}")
                st.metric("Spread accuracy (ATS vs spread_close)", f"{eval_results['spread_accuracy']:.2%}")

                # Detailed comparison table
                cols = [
                    "date", "home_team", "away_team",
                    "home_score", "away_score",
                    "predicted_home_score", "predicted_away_score",
                    "predicted_total", "total_close",
                    "predicted_margin", "spread_close",
                    "predicted_winner", "actual_winner",
                    "predicted_total_side", "actual_total_side",
                    "predicted_spread_cover", "actual_spread_cover"
                ]
                display_cols = [c for c in cols if c in eval_results["df"].columns]
                st.dataframe(eval_results["df"][display_cols])

                # Save and plot accuracy trends
                history = save_accuracy_trends(eval_results, sport=sport_key)
                st.subheader("📈 Accuracy trends over time")
                try:
                    chart_df = history.set_index("timestamp")[["moneyline_accuracy", "total_accuracy", "spread_accuracy"]]
                    st.line_chart(chart_df)
                except Exception:
                    st.info("Not enough history yet to plot trends.")
            except Exception as e:
                st.error(f"Retraining failed: {e}")
