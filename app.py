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

st.set_page_config(page_title="Sports Betting EV + ML Model", layout="wide")
st.title("📈 Sports Betting EV Model with ML Integration")

# Sidebar
st.sidebar.header("TheOddsAPI Settings")
api_key = st.sidebar.text_input("API Key", type="password")
sport_key = st.sidebar.selectbox(
    "Sport",
    ["baseball_mlb", "basketball_nba", "americanfootball_nfl", "icehockey_nhl"]
)
region = st.sidebar.selectbox("Region", ["us", "us2", "eu", "uk"])
btn_fetch = st.sidebar.button("Fetch Live Odds")

st.sidebar.header("Model management")
days_back = st.sidebar.selectbox("Scores days back (1–3)", options=[1, 2, 3], index=0)
btn_retrain = st.sidebar.button("Retrain model with merged scores+odds")

# -------------------------
# Helper functions
# -------------------------
def fetch_live_odds(api_key, sport, region, odds_format="american"):
    markets = "h2h,spreads,totals"
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds?apiKey={api_key}&regions={region}&markets={markets}&oddsFormat={odds_format}"
    r = requests.get(url)
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
        return pd.DataFrame(columns=["game_id","home_team","away_team","spread_close","total_close"])
    df = odds_df.copy()
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    teams = df.groupby("game_id", as_index=False).agg(home_team=("home_team","first"), away_team=("away_team","first"))
    totals = df[df["market"]=="totals"].dropna(subset=["line"])
    if not totals.empty:
        totals_mean = totals.groupby("game_id", as_index=False)["line"].mean().rename(columns={"line":"total_close"})
        totals_df = teams.merge(totals_mean,on="game_id",how="left")
    else:
        totals_df = teams.copy(); totals_df["total_close"]=np.nan
    spreads = df[df["market"]=="spreads"].dropna(subset=["line"])
    if not spreads.empty:
        spreads_mean = spreads.groupby(["game_id","team"],as_index=False)["line"].mean()
        pivot = spreads_mean.pivot(index="game_id",columns="team",values="line").reset_index()
        spreads_sel = teams.merge(pivot,on="game_id",how="left")
        def pick_home_spread(row):
            ht=row["home_team"]
            if ht in row.index and pd.notna(row.get(ht)): return row.get(ht)
            vals=[float(v) for k,v in row.items() if k not in ("game_id","home_team","away_team") and pd.notna(v)]
            return np.mean(vals) if vals else np.nan
        spreads_sel["spread_close"]=spreads_sel.apply(pick_home_spread,axis=1)
        spreads_df=spreads_sel[["game_id","home_team","away_team","spread_close"]]
    else:
        spreads_df=teams.copy(); spreads_df["spread_close"]=np.nan
    out=spreads_df.merge(totals_df[["game_id","total_close"]],on="game_id",how="left")
    return out

def fetch_scores(api_key,sport="basketball_nba",days_back=1):
    days_back=max(1,min(int(days_back),3))
    url=f"https://api.the-odds-api.com/v4/sports/{sport}/scores?apiKey={api_key}&daysFrom={days_back}"
    r=requests.get(url)
    if r.status_code!=200:
        st.error(f"Error fetching scores: {r.status_code} - {r.text}")
        return pd.DataFrame()
    rows=[]
    for game in r.json():
        if not game.get("completed"): continue
        home,away=game.get("home_team"),game.get("away_team")
        scores=game.get("scores",[])
        home_score=away_score=None
        if isinstance(scores,list):
            try:
                s_home=next((s for s in scores if s.get("name")==home),None)
                s_away=next((s for s in scores if s.get("name")==away),None)
                home_score=int(s_home.get("score")) if s_home and s_home.get("score") else None
                away_score=int(s_away.get("score")) if s_away and s_away.get("score") else None
            except: pass
        if home_score is None or away_score is None: continue
        rows.append({"game_id":game.get("id"),"date":game.get("commence_time"),
                     "home_team":home,"away_team":away,
                     "home_score":home_score,"away_score":away_score})
    return pd.DataFrame(rows)

def build_training_frame(api_key,sport,region,days_back=1):
    scores_df=fetch_scores(api_key,sport,days_back)
    if scores_df.empty: return pd.DataFrame()
    odds_df=fetch_live_odds(api_key,sport,region)
    if odds_df.empty: return pd.DataFrame()
    lines_df=extract_game_lines(odds_df)
    if lines_df.empty: return pd.DataFrame()
    merged=scores_df.merge(lines_df,on="game_id",how="left")
    for c in ["home_score","away_score","spread_close","total_close"]:
        merged[c]=pd.to_numeric(merged[c],errors="coerce")
    merged=merged.dropna(subset=["home_score","away_score"])
    return merged

def retrain_and_log(df,sport="basketball_nba"):
    df=df.copy()
    for sc in ["home_score","away_score"]:
        df[sc]=pd.to_numeric(df[sc],errors="coerce")
    df=df.dropna(subset=["home_score","away_score"])
    for col in ["spread_close","total_close"]:
        if col not in df.columns: df[col]=np.nan
        df[col]=pd.to_numeric(df[col],errors="coerce")
    df=df.dropna(subset=["spread_close","total_close"],how="all")
    if df.empty: raise ValueError("No training rows with spread or total present.")
    X=df[["spread_close","total_close"]].astype(float)
    y=(df["home_score"].values>df["away_score"].values).astype(int)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=GradientBoostingClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1]
    metrics={"accuracy":float(accuracy_score(y_test,y_pred)),
             "brier_score":float(brier_score_loss(y_test,y_prob)),
             "log_loss":float(log_loss(y_test,y_prob))}
    joblib.dump(model,f"{sport}_model.pkl")
    return model,X_test,y_test,y_prob,metrics

def plot_calibration_curve(y_true,y_prob):
    prob_true,prob_pred=calibration_curve(y_true,y_prob,n_bins=10,strategy="uniform")
    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot(prob_pred,prob_true,marker='o',label='Model')
    ax.plot([0,1],[0,1],linestyle='--',color='gray',label='Perfect Calibration')
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve"); ax.legend()
    st.pyplot(fig)

# -------------------------
# Streamlit actions
# -------------------------
if btn_fetch and api_key:
    odds_df=fetch_live_odds(api_key,sport_key,region)
    if odds_df.empty:
        st.warning("No live odds returned.")
    else:
        st.subheader("🔍 Raw Odds")
        st.dataframe(odds_df)

if btn_retrain and api_key:
    with st.spinner("Fetching scores + odds and retraining..."):
        df_train = build_training_frame(api_key, sport_key


