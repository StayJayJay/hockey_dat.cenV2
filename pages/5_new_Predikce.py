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
# TRAIN ML MODEL
# ==================================================
features = [
    "Home",
    "PP_Diff",
    "Goalie_rating",
    "Team_strength",
    "quality",
    "Team_form",
    "Opponent_form"
]

X = df[features]
y = df["Win"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

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
        "Opponent_form": opp_form
    }])

    prob = model.predict_proba(X_input)[0][1]

    st.metric("📊 Pravděpodobnost výhry", f"{prob*100:.1f} %")