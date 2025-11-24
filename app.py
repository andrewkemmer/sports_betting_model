def fetch_live_odds(api_key, sport, region, odds_format="american"):
    # Always request h2h, spreads, and totals together
    markets = "h2h,spreads,totals"
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        f"?apiKey={api_key}&regions={region}&markets={markets}&oddsFormat={odds_format}"
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

def extract_game_lines(df):
    # Spread: average line across bookmakers
    spreads = df[df["market"] == "spreads"].groupby("game_id")["line"].mean().rename("spread_close")
    # Total: average line across bookmakers
    totals = df[df["market"] == "totals"].groupby("game_id")["line"].mean().rename("total_close")

    games = df.groupby("game_id").agg({
        "home_team": "first",
        "away_team": "first"
    }).reset_index()

    games = games.merge(spreads.reset_index(), on="game_id", how="left")
    games = games.merge(totals.reset_index(), on="game_id", how="left")
    return games

def predict_scores_from_lines(df, model):
    df = df.copy()
    # Only compute predictions if both spread and total are present
    mask = df["spread_close"].notna() & df["total_close"].notna()
    if mask.any():
        X = df.loc[mask, ["spread_close", "total_close"]].fillna(0)
        df.loc[mask, "model_prob_home_win"] = model.predict_proba(X)[:, 1]

        df.loc[mask, "predicted_margin"] = df.loc[mask, "spread_close"] * (
            2 * df.loc[mask, "model_prob_home_win"] - 1
        )
        df.loc[mask, "predicted_total"] = df.loc[mask, "total_close"]

        df.loc[mask, "predicted_home_score"] = (
            df.loc[mask, "predicted_total"] + df.loc[mask, "predicted_margin"]
        ) / 2
        df.loc[mask, "predicted_away_score"] = (
            df.loc[mask, "predicted_total"] - df.loc[mask, "predicted_home_score"]
        )

        df.loc[mask, "predicted_home_score"] = df.loc[mask, "predicted_home_score"].round(1)
        df.loc[mask, "predicted_away_score"] = df.loc[mask, "predicted_away_score"].round(1)
        df.loc[mask, "predicted_total"] = df.loc[mask, "predicted_total"].round(1)
    return df
