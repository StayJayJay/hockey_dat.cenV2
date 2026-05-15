# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Predikce", layout="wide")
st.title("🎯 Predikce zápasu")

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    base_path = Path(__file__).resolve().parent.parent
    file_path = base_path / "data" / "HOCKEY_LOGIC_PREDICTIONS.xlsx"

    return pd.read_excel(file_path, sheet_name="MODEL_INPUT")


df = load_data()

# ==================================================
# FEATURE ENGINEERING (STEJNÉ JAKO VALIDACE)
# ==================================================
df = df.sort_values(by="Date")

df["Team_form"] = (
    df.groupby("Team")["Win"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

team_form_lookup = df[["Team", "Date", "Team_form"]].copy()
team_form_lookup.columns = ["Opponent", "Date", "Opponent_form"]

df = df.merge(team_form_lookup, on=["Opponent", "Date"], how="left")
df["Opponent_form"] = df["Opponent_form"].fillna(0.5)

df["quality"] = df["xG_Diff_adj"]
mask = df["quality"].isna() | (df["quality"] == 0)
df.loc[mask, "quality"] = df["Shots_Diff"] * 0.1

# ==================================================
# VYPÍTEJ H2H PRO CELÝ DATASET (DŮLEŽITÉ)
# ==================================================

def compute_h2h_for_df(df):
    h2h_list = []

    for idx, row in df.iterrows():
        team = row["Team"]
        opponent = row["Opponent"]

        past_matches = df.iloc[:idx]

        h2h = past_matches[
            ((past_matches["Team"] == team) & (past_matches["Opponent"] == opponent)) |
            ((past_matches["Team"] == opponent) & (past_matches["Opponent"] == team))
        ]

        if len(h2h) == 0:
            h2h_list.append(0.5)
            continue

        h2h_last = h2h.tail(3)

        wins = 0
        total = 0

        for _, m in h2h_last.iterrows():
            if m["Team"] == team:
                wins += m["Win"]
            else:
                wins += (1 - m["Win"])
            total += 1

        h2h_list.append(wins / total if total > 0 else 0.5)

    return h2h_list


df["H2H_form"] = compute_h2h_for_df(df)

# ==================================================
# TRAIN ML MODEL
# ==================================================
features = [
    "Home",
    "PP_Diff",
    "Goalie_rating",
    "Team_strength",
    "quality",
    "Team_form",
    "Opponent_form",
    "H2H_form"
]

X = df[features]
y = df["Win"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ==================================================
# HEAD-TO-HEAD FORM (last 3 games)
# ==================================================
def get_h2h_form(team, opponent, n_games=3):
    
    # vyfiltruj vzájemné zápasy
    h2h = df[
        ((df["Team"] == team) & (df["Opponent"] == opponent)) |
        ((df["Team"] == opponent) & (df["Opponent"] == team))
    ].sort_values("Date")

    if len(h2h) == 0:
        return 0.5

    # vezmi poslední n zápasů
    h2h_last = h2h.tail(n_games)

    # přepočti z pohledu teamu
    wins = 0
    total = 0

    for _, row in h2h_last.iterrows():
        if row["Team"] == team:
            wins += row["Win"]
        else:
            # když je team "Opponent", musíš invertovat Win
            wins += (1 - row["Win"])
        
        total += 1

    return wins / total if total > 0 else 0.5

# ==================================================
# UI INPUT
# ==================================================
st.subheader("Zadej zápas")

team = st.selectbox("Team", sorted(df["Team"].unique()))
opponent = st.selectbox("Opponent", sorted(df["Team"].unique()))

home = st.selectbox("Home (1 = doma)", [1, 0])

pp = st.number_input("PP Diff", value=0.0)
goalie = st.number_input("Goalie rating", value=0.0)
strength = st.number_input("Team strength", value=0.0)

# vezmi historickou formu
team_form = df[df["Team"] == team]["Team_form"].iloc[-1]
opp_form = df[df["Team"] == opponent]["Team_form"].iloc[-1]

# quality fallback (zadáš xG nebo jen použij default)
quality = st.number_input("xG Diff (nebo 0)", value=0.0)

team_data = get_latest_team_data(team)
opp_data = get_latest_team_data(opponent)

h2h_form = get_h2h_form(team, opponent)

st.write(f"🔥 H2H poslední 3 zápasy: {h2h_form:.2f}")

# ==================================================
# PREDIKCE
# ==================================================
if st.button("Spočítat predikci"):

    X_input = pd.DataFrame([{
        "Home": home,
        "PP_Diff": pp,
        "Goalie_rating": goalie,
        "Team_strength": strength,
        "quality": quality,
        "Team_form": team_form,
        "Opponent_form": opp_form,
        "H2H_form": h2h_form
    }])

    prob = model.predict_proba(X_input)[0][1]

    st.metric("📊 Pravděpodobnost výhry", f"{prob*100:.1f} %")