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
# HELPER – poslední data týmu
# ==================================================
def get_latest_team_data(team_name):
    data = df[df["Team"] == team_name].sort_values("Date")

    if len(data) == 0:
        return {
            "PP_Diff": 0.0,
            "Goalie_rating": 0.0,
            "Team_strength": 0.0,
            "Team_form": 0.5
        }

    last_row = data.iloc[-1]

    return {
        "PP_Diff": last_row.get("PP_Diff", 0.0),
        "Goalie_rating": last_row.get("Goalie_rating", 0.0),
        "Team_strength": last_row.get("Team_strength", 0.0),
        "Team_form": last_row.get("Team_form", 0.5)
    }

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
# Batch
# ==================================================

st.subheader("🚀 Batch predikce (celé kolo)")

today = pd.Timestamp.today().normalize()
df_today = df[df["Date"] == today]

if df_today.empty:
    st.warning("⚠️ Dnes nejsou žádné zápasy v datasetu")
    st.stop()


results = []

for _, row in df_today.iterrows():

    X_row = pd.DataFrame([{
        "Home": row["Home"],
        "PP_Diff": row["PP_Diff"],
        "Goalie_rating": row["Goalie_rating"],
        "Team_strength": row["Team_strength"],
        "quality": row["quality"],
        "Team_form": row["Team_form"],
        "Opponent_form": row["Opponent_form"],
        "H2H_form": row["H2H_form"]
    }])

    prob = model.predict_proba(X_row)[0][1]

    # ✅ kalibrace
    prob_cal = 0.5 + (prob - 0.5) * 0.6

    # ✅ kurz (zatím fake, později napojíš real odds)
    odds = 2.0

    implied = 1 / odds
    edge = prob_cal - implied
    ev = (prob_cal * odds) - 1

    results.append({
        "Team": row["Team"],
        "Opponent": row["Opponent"],
        "Prob": prob_cal,
        "Odds": odds,
        "Edge": edge,
        "EV": ev
    })

df_results = pd.DataFrame(results)
df_value = df_results[df_results["Edge"] > 0.05]
df_value = df_value.sort_values("Edge", ascending=False)


# ==================================================
# PREDIKCE
# ==================================================

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

# kalibrace (shrinking extrémů)
prob_calibrated = 0.5 + (prob - 0.5) * 0.6

st.metric("📊 Pravděpodobnost výhry", f"{prob_calibrated*100:.1f} %")

if prob > 0.65:
    st.success("🔥 Strong pick")
elif prob > 0.55:
    st.info("✅ Slight advantage")
else:
    st.warning("⚖️ No clear edge")


if team_form > opp_form:
    st.success("✅ Tým je ve lepší formě")
else:
    st.warning("⚠️ Soupeř má lepší formu")

st.subheader("🔍 Vysvětlení")

st.write(f"📊 Forma týmu: {team_form:.2f}")
st.write(f"📊 Forma soupeře: {opp_form:.2f}")
st.write(f"🔥 H2H: {h2h_form:.2f}")
st.write(f"⚡ PP diff: {pp:.2f}")

odds = st.number_input("Kurz (např. 1.90)", value=2.0)
implied_prob = 1 / odds
prob_calibrated = 0.5 + (prob - 0.5) * 0.6

st.subheader("💰 Value analýza")

st.write(f"📊 Model pravděpodobnost: {prob_calibrated*100:.1f} %")
st.write(f"📊 Implied (kurz): {implied_prob*100:.1f} %")

edge = prob_calibrated - implied_prob

if edge > 0.05:
    st.success(f"💰 VALUE BET (+{edge*100:.1f} % edge)")
elif edge > 0:
    st.info(f"✅ Malá value (+{edge*100:.1f} %)")
else:
    st.error(f"❌ No value ({edge*100:.1f} %)")

ev = (prob_calibrated * odds) - 1

st.write(f"📈 Expected Value (EV): {ev:.2f}")

st.subheader("💰 Nejlepší zápasy (VALUE)")

st.dataframe(
    df_value[["Team", "Opponent", "Prob", "Odds", "Edge", "EV"]]
    .style.format({
        "Prob": "{:.2%}",
        "Edge": "{:.2%}",
        "EV": "{:.2f}"
    })
)

def highlight(row):
    if row.Edge > 0.1:
        return ["background-color: lightgreen"] * len(row)
    return [""] * len(row)

st.dataframe(df_value.style.apply(highlight, axis=1))
