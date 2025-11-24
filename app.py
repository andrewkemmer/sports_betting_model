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
from sklearn.impute import SimpleImputer
from pathlib import Path

st.set_page_config(page_title="Sports Betting Model", layout="wide")
st.title("📈 Sports Betting Model — Predictions & Accuracy")

# Sidebar
st.sidebar.header("TheOddsAPI Settings")
api_key = st.sidebar.text_input("API Key", type="password")
sport_key = st.sidebar.selectbox(
    "Sport",
    ["basketball_nba", "baseball_mlb", "americanfootball_nfl", "icehockey_nhl"],
)
region = st.sidebar.selectbox("Region", ["us", "us2", "eu", "uk"])
btn_fetch = st.sidebar.button("Fetch live odds")
st.sidebar.header("Model management")

# Scores endpoint limits: The Odds API accepts daysFrom typically between 1 and 3
days_back = st.sidebar.selectbox("Scores days back (1-3)", options=[1, 2, 3], index=0)
btn_retrain = st.sidebar.button("Retrain model (use scores daysBack)")
st.sidebar.markdown("Imputation for missing features")
impute_option = st.sidebar.selectbox("Impute strategy", ["none (drop rows)", "median"])
impute_key = "median" if impute_option == "median" else "none"

DEBUG = st.sidebar.checkbox("Show prediction debug", value=False)

MODEL_DIR = Path(".")
MODEL_PATH = lambda sport: MODEL_DIR / f"{sport}_model.pkl"
IMPUTER_PATH = lambda sport: MODEL_DIR / f"{sport}_imputer.pkl"

# -------------------------
# Odds helpers
# -------------------------
def fetch_live_odds(api_key, sport, region, odds_format="american"):
    markets = "h2h,spreads,totals"
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds?apiKey={api_key}&regions={region}&markets={markets}&oddsFormat={odds_format}"
    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        st.error(f"Error fetching odds: {e}")
        return pd.DataFrame()
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
                    rows.append(
                        {
                            "game_id": gid,
                            "market": mk.get("key"),
                            "team": o.get("name"),
                            "line": o.get("point"),
                            "home_team": home,
                            "away_team": away,
                        }
                    )
    return pd.DataFrame(rows)


def extract_game_lines(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(columns=["game_id", "home_team", "away_team", "spread_close", "total_close"])

    df = odds_df.copy()
    market_map = {
        "spread": "spreads",
        "point_spread": "spreads",
        "spreads_market": "spreads",
        "over_under": "totals",
        "ou": "totals",
        "totals_market": "totals",
    }
    df["market"] = df["market"].map(lambda x: market_map.get(x, x))
    for c in ["game_id", "market", "team", "line", "home_team", "away_team"]:
        if c not in df.columns:
            df[c] = np.nan
    df["line"] = pd.to_numeric(df["line"], errors="coerce")

    teams = df.groupby("game_id", as_index=False).agg(home_team=("home_team", "first"), away_team=("away_team", "first"))

    totals = df[df["market"] == "totals"].dropna(subset=["line"])
    if not totals.empty:
        totals_mean = totals.groupby("game_id", as_index=False)["line"].mean().rename(columns={"line": "total_close"})
        totals_df = teams.merge(totals_mean, on="game_id", how="left")
    else:
        totals_df = teams.copy()
        totals_df["total_close"] = np.nan

    spreads = df[df["market"] == "spreads"].dropna(subset=["line"])
    if not spreads.empty:
        spreads_mean = spreads.groupby(["game_id", "team"], as_index=False)["line"].mean()
        pivot = spreads_mean.pivot(index="game_id", columns="team", values="line").reset_index()
        spreads_sel = teams.merge(pivot, on="game_id", how="left")

        def pick_home_spread(row):
            ht = row["home_team"]
            if pd.notna(ht) and ht in row.index and pd.notna(row.get(ht)):
                return row.get(ht)
            numeric_vals = []
            for k, v in row.items():
                if k not in ("game_id", "home_team", "away_team") and pd.notna(v):
                    try:
                        numeric_vals.append(float(v))
                    except Exception:
                        pass
            if numeric_vals:
                return float(np.mean(numeric_vals))
            return np.nan

        spreads_sel["spread_close"] = spreads_sel.apply(pick_home_spread, axis=1)
        spreads_df = spreads_sel[["game_id", "home_team", "away_team", "spread_close"]]
    else:
        spreads_df = teams.copy()
        spreads_df["spread_close"] = np.nan

    out = spreads_df.merge(totals_df[["game_id", "total_close"]], on="game_id", how="left")
    out["spread_close"] = pd.to_numeric(out["spread_close"], errors="coerce")
    out["total_close"] = pd.to_numeric(out["total_close"], errors="coerce")
    return out


# -------------------------
# Scores helpers (completed games only)
# -------------------------
def fetch_scores_with_odds(api_key, sport="basketball_nba", days_back=1):
    # Clamp days_back to allowed range for the scores endpoint (1..3)
    try:
        days_back = int(days_back)
    except Exception:
        days_back = 1
    days_back = max(1, min(days_back, 3))

    url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores?apiKey={api_key}&daysFrom={days_back}"
    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        st.error(f"Error fetching scores: {e}")
        return pd.DataFrame()
    if r.status_code != 200:
        st.error(f"Error fetching scores: {r.status_code} - {r.text}")
        return pd.DataFrame()
    rows = []
    for game in r.json():
        if not game.get("completed"):
            continue
        home, away = game.get("home_team"), game.get("away_team")
        date = game.get("commence_time")
        scores = game.get("scores", [])
        home_score, away_score = None, None
        if isinstance(scores, list):
            try:
                s_home = next((s for s in scores if s.get("name") == home), None)
                s_away = next((s for s in scores if s.get("name") == away), None)
                home_score = s_home.get("score") if s_home and s_home.get("score") is not None else None
                away_score = s_away.get("score") if s_away and s_away.get("score") is not None else None
            except Exception:
                pass
        if home_score is None or away_score is None:
            continue
        rows.append(
            {
                "game_id": game.get("id"),
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "spread_close": None,
                "total_close": None,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "game_id" in df.columns and df["game_id"].notna().any():
        df = df.drop_duplicates(subset=["game_id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["home_team", "away_team", "date"], keep="last")
    return df


# -------------------------
# Model training
# -------------------------
def retrain_and_log(df: pd.DataFrame, sport="basketball_nba", impute="none"):
    df = df.copy()

    # Coerce scores to numeric and drop invalid rows
    for sc in ["home_score", "away_score"]:
        if sc not in df.columns:
            df[sc] = np.nan
        df[sc] = pd.to_numeric(df[sc], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        raise ValueError("No completed games found to retrain after cleaning scores.")

    # Ensure line columns exist and numeric
    for col in ["spread_close", "total_close"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Require at least one feature per row
    df = df.dropna(subset=["spread_close", "total_close"], how="all")
    if df.empty:
        raise ValueError("No training rows with spread or total present. Acquire historical lines before retraining.")

    X = df[["spread_close", "total_close"]].astype(float)
    y = (df["home_score"] > df["away_score"]).astype(int)

    # Time-based split if date exists
    if "date" in df.columns:
        df_sorted = df.assign(date=pd.to_datetime(df["date"], errors="coerce")).sort_values("date")
        cut = int(len(df_sorted) * 0.8)
        train_idx = df_sorted.index[:cut]
        test_idx = df_sorted.index[cut:]
        X_train, X_test = X.loc[train_idx].copy(), X.loc[test_idx].copy()
        y_train, y_test = y.loc[train_idx].copy(), y.loc[test_idx].copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = None
    if impute == "median":
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)
    else:
        X_train = X_train.dropna(how="all")
        y_train = y_train.loc[X_train.index]
        X_test = X_test.dropna(how="all")
        y_test = y_test.loc[X_test.index]

    if X_train.empty or X_test.empty:
        raise ValueError("Insufficient non-missing feature rows after preprocessing for training/testing.")

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    X_test_eval = X_test[["spread_close", "total_close"]].astype(float)
    y_pred = model.predict(X_test_eval)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Trained model does not implement predict_proba.")
    y_prob = model.predict_proba(X_test_eval)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test.loc[X_test_eval.index], y_pred)),
        "brier_score": float(brier_score_loss(y_test.loc[X_test_eval.index], y_prob)),
        "log_loss": float(log_loss(y_test.loc[X_test_eval.index], y_prob)),
    }

    try:
        joblib.dump(model, str(MODEL_PATH(sport)))
        if imputer is not None:
            joblib.dump(imputer, str(IMPUTER_PATH(sport)))
    except Exception as e:
        st.warning(f"Failed to persist model/imputer to disk: {e}")

    return model, X_test_eval, y_test.loc[X_test_eval.index], y_prob, metrics


# -------------------------
# Predictions
# -------------------------
def load_model_and_imputer(sport):
    model = None
    imputer = None
    try:
        if MODEL_PATH(sport).exists():
            model = joblib.load(str(MODEL_PATH(sport)))
    except Exception:
        model = None
    try:
        if IMPUTER_PATH(sport).exists():
            imputer = joblib.load(str(IMPUTER_PATH(sport)))
    except Exception:
        imputer = None
    return model, imputer


def predict_scores_from_lines(df: pd.DataFrame, model, imputer=None, impute="none"):
    df = df.copy()
    for c in ["spread_close", "total_close"]:
        if c not in df.columns:
            df[c] = np.nan
    df["spread_close"] = pd.to_numeric(df["spread_close"], errors="coerce")
    df["total_close"] = pd.to_numeric(df["total_close"], errors="coerce")

    mask_spread = df[["spread_close", "total_close"]].notna().any(axis=1)
    df["predicted_margin"] = np.nan
    df["model_prob_home_win"] = np.nan
    df["model_prob_away_win"] = np.nan

    if mask_spread.any():
        feature_cols = ["spread_close", "total_close"]
        X_raw = df.loc[mask_spread, feature_cols].astype(float)

        if impute == "median":
            if imputer is not None:
                X_valid = pd.DataFrame(imputer.transform(X_raw), index=X_raw.index, columns=X_raw.columns)
            else:
                tmp_imp = SimpleImputer(strategy="median")
                X_valid = pd.DataFrame(tmp_imp.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)
        else:
            X_valid = X_raw.dropna(how="all")

        if X_valid.empty:
            st.warning("No valid feature rows to predict after dropping missing values.")
            return df

        X_valid = X_valid[feature_cols]

        if DEBUG:
            st.write("DEBUG: prediction input shape:", X_valid.shape)
            st.write("DEBUG: prediction input head:", X_valid.head(6))
            st.write("DEBUG: prediction input nunique:", X_valid.nunique())
            st.write("DEBUG: prediction input describe:", X_valid.describe())

        if model is None or not hasattr(model, "predict_proba"):
            st.error("Model does not support predict_proba or is not available. Retrain with a classifier that implements predict_proba.")
            return df

        try:
            probs = model.predict_proba(X_valid)[:, 1]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            return df

        df.loc[X_valid.index, "model_prob_home_win"] = probs
        df.loc[X_valid.index, "model_prob_away_win"] = 1 - probs

        spread_vals = df.loc[X_valid.index, "spread_close"].fillna(0.0)
        df.loc[X_valid.index, "predicted_margin"] = spread_vals * (0.5 + df.loc[X_valid.index, "model_prob_home_win"])

        if np.isclose(probs.max(), probs.min()):
            st.warning("All predicted probabilities are identical — check training data variance and input feature values.")
            if DEBUG:
                st.write("Unique probs:", np.unique(np.round(probs, 6)))
                st.write("Input describe:", X_valid.describe())

    mask_total = df["total_close"].notna() & df["predicted_margin"].notna()
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
def plot_calibration(y_true, y_prob):
    fig, ax = plt.subplots()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability")
    ax.legend()
    st.pyplot(fig)


# -------------------------
# Streamlit actions
# -------------------------
if btn_fetch and api_key:
    odds_df = fetch_live_odds(api_key, sport_key, region)
    if not odds_df.empty:
        lines_df = extract_game_lines(odds_df)
        model, imputer = load_model_and_imputer(sport_key)
        if model is None:
            st.warning("No trained model found. Please retrain first.")
            model = None

        if model is not None:
            preds = predict_scores_from_lines(lines_df, model, imputer=imputer, impute=impute_key)
            st.subheader("Predicted Scores")
            st.dataframe(
                preds[
                    [
                        "home_team",
                        "away_team",
                        "predicted_home_score",
                        "predicted_away_score",
                        "predicted_total",
                        "model_prob_home_win",
                        "model_prob_away_win",
                    ]
                ]
            )
        else:
            st.info("Model not available yet.")
    else:
        st.warning("No odds returned from API. Check API key, sport, and region.")

if btn_retrain and api_key:
    with st.spinner("Fetching historical scores and retraining..."):
        scores_df = fetch_scores_with_odds(api_key, sport_key, days_back=days_back)
        if not scores_df.empty:
            try:
                # If scores endpoint returns no lines, consider fetching odds separately and merging on game_id
                model, X_test, y_test, y_prob, metrics = retrain_and_log(
                    scores_df, sport_key, impute=impute_key
                )
                st.success("Model retrained successfully!")
                st.json(metrics)
                plot_calibration(y_test, y_prob)
            except Exception as e:
                st.error(f"Retraining failed: {e}")
        else:
            st.warning("No completed games found for retraining. If the scores endpoint returns no lines, fetch odds and merge on game_id to supply spread_close/total_close.")


