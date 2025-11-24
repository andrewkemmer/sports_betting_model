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
# Helpers: scores
# -------------------------
def fetch_scores_with_odds(api_key, sport="basketball_nba", days_back=30) -> pd.DataFrame:
    rows = []
    for d in range(1, days_back + 1):
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores/?apiKey={api_key}&daysFrom={d}"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            continue
        for game in r.json():
            home = game.get("home_team")
            away = game.get("away_team")
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
            rows.append({
                "date": game.get("commence_time"),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "spread_close": None,
                "total_close": None
            })
    return pd.DataFrame(rows)

# -------------------------
# Model training (home win probability classifier)
# -------------------------
def retrain_and_log(df: pd.DataFrame, sport="basketball_nba"):
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
    # Ensure required columns exist
    if "spread_close" not in df.columns:
        df["spread_close"] = np.nan
    if "total_close" not in df.columns:
        df["total_close"] = np.nan

    # Compute win probabilities using classifier
    mask_spread = df["spread_close"].notna()
    df["predicted_margin"] = np.nan
    df["model_prob_home_win"] = np.nan
    df["model_prob_away_win"] = np.nan

    if mask_spread.any():
        X = df.loc[mask_spread, ["spread_close", "total_close"]].fillna(0)
        probs = model.predict_proba(X)[:, 1]
        df.loc[mask_spread, "model_prob_home_win"] = probs
        df.loc[mask_spread, "model_prob_away_win"] = 1 - probs
        # Margin heuristic: keep signal even when prob ~ 0.5
        df.loc[mask_spread, "predicted_margin"] = df.loc[mask_spread, "spread_close"] * (0.5 + probs)

    # Compute scores only when totals are present
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
            df.loc[mask_total, "predicted_total"] - df.loc[mask_total, "predicted_margin"]
        ) / 2
        df.loc[mask_total, "predicted_home_score"] = df.loc[mask_total, "predicted_home_score"].round(1)
        df.loc[mask_total, "predicted_away_score"] = df.loc[mask_total, "predicted_away_score"].round(1)
        df.loc[mask_total, "predicted_total"] = df.loc[mask_total, "predicted_total"].round(1)

    return df

# -------------------------
# Evaluation
# -------------------------
def evaluate_predictions(df: pd.DataFrame, model):
    df = predict_scores_from_lines(df, model)

    # Coerce numeric columns to floats to avoid NoneType comparisons
    for col in [
        "home_score", "away_score", "spread_close", "total_close",
        "predicted_home_score", "predicted_away_score", "predicted_total", "predicted_margin"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Moneyline accuracy
    df["predicted_winner"] = np.where(
        (df["predicted_home_score"].notna()) & (df["predicted_away_score"].notna()) &
        (df["predicted_home_score"] >= df["predicted_away_score"]),
        df["home_team"], df["away_team"]
    )
    df["actual_winner"] = np.where(
        (df["home_score"].notna()) & (df["away_score"].notna()) &
        (df["home_score"] > df["away_score"]),
        df["home_team"], df["away_team"]
    )
    moneyline_mask = df["predicted_winner"].notna() & df["actual_winner"].notna()
    moneyline_acc = float((df.loc[moneyline_mask, "predicted_winner"] == df.loc[moneyline_mask, "actual_winner"]).mean()) if moneyline_mask.any() else np.nan

    # Total (Over/Under) accuracy
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["predicted_total_side"] = np.where(
        (df["predicted_total"].notna()) & (df["total_close"].notna()) &
        (df["predicted_total"] > df["total_close"]), "Over", "Under"
    )
    df["actual_total_side"] = np.where(
        (df["actual_total"].notna()) & (df["total_close"].notna()) &
        (df["actual_total"] > df["total_close"]), "Over", "Under"
    )
    total_mask = df["predicted_total_side"].notna() & df["actual_total_side"].notna()
    total_acc = float((df.loc[total_mask, "predicted_total_side"] == df.loc[total_mask, "actual_total_side"]).mean()) if total_mask.any() else np.nan

    # Spread (ATS) accuracy
    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["predicted_spread_cover"] = np.where(
        (df["predicted_margin"].notna()) & (df["spread_close"].notna()) &
        (df["predicted_margin"] > df["spread_close"]), "Home", "Away"
    )
    df["actual_spread_cover"] = np.where(
        (df["actual_margin"].notna()) & (df["spread_close"].notna()) &
        (df["actual_margin"] > df["spread_close"]), "Home", "Away"
    )
    spread_mask = df["predicted_spread_cover"].notna() & df["actual_spread_cover"].notna()
    spread_acc = float((df.loc[spread_mask, "predicted_spread_cover"] == df.loc[spread_mask, "actual_spread_cover"]).mean()) if spread_mask.any() else np.nan

    return {
        "moneyline_accuracy": moneyline_acc,
        "total_accuracy": total_acc,
        "spread_accuracy": spread_acc,
        "df": df
    }

# -------------------------
# History and plots
# -------------------------
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

# -------------------------
# Live odds button handler
# -------------------------
if btn_fetch:
    if not api_key:
        st.warning("Please enter your API key.")
    else:
        odds_df = fetch_live_odds(api_key, sport_key, region)
        if odds_df.empty:
            st.warning("No live odds returned.")
        else:
            st.subheader("🔍 Raw odds")
            st.dataframe(odds_df)
            st.write("Markets available:", odds_df["market"].unique())

            # Try to load trained model
            try:
                model = joblib.load(f"{sport_key}_model.pkl")
            except Exception:
                model = None
                st.info("No trained model found yet. Retrain to enable predictions.")

            if model is not None:
                games = extract_game_lines(odds_df)
                if games.empty:
                    st.info("No spread/total lines found for current games.")
                else:
                    pred_live = predict_scores_from_lines(games, model)
                    pred_live["home_win_pct"] = (pred_live["model_prob_home_win"] * 100).round(1)
                    pred_live["away_win_pct"] = (pred_live["model_prob_away_win"] * 100).round(1)

                    st.subheader("🧮 Live predicted scores, margins, and win probabilities")
                    st.dataframe(pred_live[[
                        "home_team", "away_team",
                        "spread_close", "total_close",
                        "predicted_margin",
                        "predicted_home_score", "predicted_away_score", "predicted_total",
                        "home_win_pct", "away_win_pct"
                    ]])

# -------------------------
# Retraining button handler
# -------------------------
if btn_retrain:
    if not api_key:
        st.warning("Please enter your API key.")
    else:
        st.info("Retraining model...")
        hist_df = fetch_scores_with_odds(api_key, sport=sport_key, days_back=30)
        if hist_df.empty:
            st.warning("No historical data retrieved.")
        else:
            try:
                model, X_test, y_test, y_prob, metrics = retrain_and_log(hist_df, sport=sport_key)
                st.success("Model retrained!")

                st.subheader("📊 Model calibration")
                st.write(f"Accuracy (test split): {metrics['accuracy']:.3f}")
                st.write(f"Brier Score: {metrics['brier_score']:.3f}")
                st.write(f"Log Loss: {metrics['log_loss']:.3f}")
                plot_calibration_curve(y_test, y_prob)

                eval_results = evaluate_predictions(hist_df, model)

                st.subheader("🏁 Accuracy vs actual outcomes (last 30 days)")
                ml = eval_results["moneyline_accuracy"]
                ou = eval_results["total_accuracy"]
                ats = eval_results["spread_accuracy"]
                st.metric("Moneyline winner accuracy", f"{(ml if pd.notna(ml) else 0):.2%}")
                st.metric("Total score accuracy (O/U)", f"{(ou if pd.notna(ou) else 0):.2%}")
                st.metric("Spread accuracy (ATS)", f"{(ats if pd.notna(ats) else 0):.2%}")

                cols = [
                    "date", "home_team", "away_team",
                    "home_score", "away_score",
                    "spread_close", "total_close",
                    "predicted_margin",
                    "predicted_home_score", "predicted_away_score", "predicted_total",
                    "predicted_winner", "actual_winner",
                    "actual_total", "predicted_total_side", "actual_total_side",
                    "actual_margin", "predicted_spread_cover", "actual_spread_cover"
                ]
                df_show = eval_results["df"].copy()
                for c in cols:
                    if c not in df_show.columns:
                        df_show[c] = np.nan
                st.dataframe(df_show[cols])

                history = save_accuracy_trends(eval_results, sport=sport_key)
                st.subheader("📈 Accuracy trends over time")
                chart_df = history.copy()
                chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
                chart_df = chart_df.set_index("timestamp")
                st.line_chart(chart_df[["moneyline_accuracy", "total_accuracy", "spread_accuracy"]])
            except Exception as e:
                st.error(f"Retraining failed: {e}")
