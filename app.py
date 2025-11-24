import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sportsipy.nba.teams import Teams
from sportsipy.nba.schedule import Schedule
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(page_title="Stats-Based Sports Prediction Model", layout="wide")
st.title("🏀 Stats-Based Sports Prediction Model (Sportsipy)")

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("Model Settings")
season = st.sidebar.number_input("Season (year)", min_value=2000, max_value=2025, value=2024)
btn_retrain = st.sidebar.button("Retrain model with Sportsipy stats")

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def fetch_games_with_stats(season=2024):
    teams = Teams(season)
    rows = []
    for team in teams:
        schedule = Schedule(team.abbreviation, season)
        for game in schedule:
            if game.points_scored is None or game.opponent_points is None:
                continue
            row = {
                "date": game.date,
                "home_team": game.opponent_abbr if game.location == "AWAY" else team.abbreviation,
                "away_team": team.abbreviation if game.location == "AWAY" else game.opponent_abbr,
                "home_score": game.opponent_points if game.location == "AWAY" else game.points_scored,
                "away_score": game.points_scored if game.location == "AWAY" else game.opponent_points,
                "home_off_rating": team.offensive_rating,
                "home_def_rating": team.defensive_rating,
                "away_off_rating": getattr(game.opponent, "offensive_rating", None),
                "away_def_rating": getattr(game.opponent, "defensive_rating", None),
                "home_recent_win_pct": team.wins / (team.wins + team.losses) if (team.wins + team.losses) > 0 else 0,
                "away_recent_win_pct": getattr(game.opponent, "wins", 0) / max(1, getattr(game.opponent, "wins", 0) + getattr(game.opponent, "losses", 0))
            }
            rows.append(row)
    return pd.DataFrame(rows)

def build_features(df):
    df = df.dropna(subset=["home_score", "away_score"])
    df["off_diff"] = df["home_off_rating"] - df["away_off_rating"]
    df["def_diff"] = df["home_def_rating"] - df["away_def_rating"]
    df["form_diff"] = df["home_recent_win_pct"] - df["away_recent_win_pct"]
    X = df[["off_diff", "def_diff", "form_diff"]].fillna(0)
    y = (df["home_score"] > df["away_score"]).astype(int)
    return X, y, df

def retrain_stats_model(season=2024, sport="basketball_nba"):
    df = fetch_games_with_stats(season)
    X, y, df = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "brier_score": brier_score_loss(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob)
    }
    joblib.dump(model, f"{sport}_stats_model.pkl")
    return model, df, metrics, y_test, y_prob

def predict_from_stats(df, model):
    X, _, df = build_features(df)
    df["model_prob_home_win"] = model.predict_proba(X)[:, 1]
    df["predicted_winner"] = df["home_team"].where(df["model_prob_home_win"] >= 0.5, df["away_team"])
    return df

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
# Retraining + accuracy dashboard
# --------------------------------------------------
if btn_retrain:
    st.info("Retraining model with Sportsipy stats...")
    try:
        model, df, metrics, y_test, y_prob = retrain_stats_model(season=season)
        st.success("Model retrained!")
        st.subheader("📊 Performance metrics (classification)")
        st.write(f"Accuracy: {metrics['accuracy']:.3f}")
        st.write(f"Brier Score: {metrics['brier_score']:.3f}")
        st.write(f"Log Loss: {metrics['log_loss']:.3f}")
        plot_calibration_curve(y_test, y_prob)

        # Predictions vs actuals
        eval_df = predict_from_stats(df, model)
        st.subheader("🏁 Predicted vs actual outcomes")
        st.dataframe(eval_df[["date", "home_team", "away_team", "home_score", "away_score", "predicted_winner", "model_prob_home_win"]])
    except Exception as e:
        st.error(f"Retraining failed: {e}")

