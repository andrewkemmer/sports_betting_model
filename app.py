
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Sports Betting EV Model", layout="wide")

st.title("📈 Sports Betting EV Model (Live Odds + No-Vig + EV Calculation)")

##############################################
# Sidebar Inputs
##############################################

st.sidebar.header("TheOddsAPI Settings")
api_key = st.sidebar.text_input("API Key", type="password")
sport_key = st.sidebar.selectbox(
    "Sport",
    ["baseball_mlb", "basketball_nba", "americanfootball_nfl", "hockey_nhl"]
)
region = st.sidebar.selectbox("Region", ["us", "us2", "eu", "uk"])
market = st.sidebar.selectbox("Market", ["h2h", "spreads", "totals"])
btn_fetch = st.sidebar.button("Fetch Live Odds")

##############################################
# Helper Functions
##############################################

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

        for book in game["bookmakers"]:
            book_name = book["title"]

            for mk in book["markets"]:
                if mk["key"] != market:
                    continue

                outcomes = mk["outcomes"]

                if market == "h2h":
                    for o in outcomes:
                        rows.append({
                            "game_id": gid,
                            "market": market,
                            "bookmaker": book_name,
                            "team": o["name"],
                            "opponent": away if o["name"] == home else home,
                            "line": None,
                            "price": o["price"]
                        })

                elif market == "spreads":
                    for o in outcomes:
                        rows.append({
                            "game_id": gid,
                            "market": market,
                            "bookmaker": book_name,
                            "team": o["name"],
                            "opponent": away if o["name"] == home else home,
                            "line": o["point"],
                            "price": o["price"]
                        })

                elif market == "totals":
                    for o in outcomes:
                        rows.append({
                            "game_id": gid,
                            "market": market,
                            "bookmaker": book_name,
                            "team": o["name"],  # Over or Under
                            "opponent": None,
                            "line": o["point"],
                            "price": o["price"]
                        })

    df = pd.DataFrame(rows)
    return df


def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def remove_vig(prob_a, prob_b):
    """Normalize two implied probabilities to remove vig."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def ev_calc(prob, odds):
    """Expected value calculation."""
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / -odds
    return prob * payout - (1 - prob)


##############################################
# Main Processing
##############################################

if btn_fetch:
    if not api_key:
        st.warning("Please enter your API key.")
    else:
        df = fetch_live_odds(api_key, sport_key, region, market)

        if df is not None:
            st.subheader("🔍 Raw Odds")
            st.dataframe(df)

            # Rename to fit EV pipeline
            df = df.rename(columns={"price": "book_american"})

            results = []

            grouped = df.groupby(["game_id", "bookmaker"])

            for (gid, book), g in grouped:
                if len(g) != 2:
                    continue

                t1, t2 = g.iloc[0], g.iloc[1]

                p1 = american_to_prob(t1.book_american)
                p2 = american_to_prob(t2.book_american)

                nv1, nv2 = remove_vig(p1, p2)

                ev1 = ev_calc(nv1, t1.book_american)
                ev2 = ev_calc(nv2, t2.book_american)

                results.append({
                    "game_id": gid,
                    "bookmaker": book,
                    "team": t1.team,
                    "opponent": t1.opponent,
                    "line": t1.line,
                    "odds": t1.book_american,
                    "no_vig_prob": nv1,
                    "EV": ev1
                })

                results.append({
                    "game_id": gid,
                    "bookmaker": book,
                    "team": t2.team,
                    "opponent": t2.opponent,
                    "line": t2.line,
                    "odds": t2.book_american,
                    "no_vig_prob": nv2,
                    "EV": ev2
                })

            out = pd.DataFrame(results)

            st.subheader("📊 No-Vig Odds + EV")
            st.dataframe(out)

            st.subheader("💰 Positive EV Bets")
            pos_ev = out[out["EV"] > 0].sort_values("EV", ascending=False)

            if pos_ev.empty:
                st.write("No positive EV bets found.")
            else:
                st.dataframe(pos_ev)
