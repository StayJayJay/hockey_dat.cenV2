# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Predikce zápasu", layout="wide")
st.title("Predikce zápasu – ELH")

# ==================================================
# Načtení dat
# ==================================================
@st.cache_data
def load_data():
    base_path = Path(__file__).resolve().parent.parent
    file_path = base_path / "data" / "HOCKEY_LOGIC_PREDICTIONS.xlsx"

    params_raw = pd.read_excel(
        file_path,
        sheet_name="PARAMETRY",
        header=1  # ← KLÍČOVÉ
    )

    teams = pd.read_excel(
        file_path,
        sheet_name="CALC_TEAMS_SEASON"
    )

    return params_raw, teams
# ==========================================
# Normalizace PARAMETRY
# ==========================================
params_raw, teams = load_data()
params_raw = params_raw.iloc[:, :4]
params_raw.columns = ["Parameter", "Coefficient", "Source", "Note"]
params_raw = params_raw.dropna(subset=["Parameter", "Coefficient"])

params_raw["Parameter"] = params_raw["Parameter"].astype(str).str.strip()

params = params_raw.set_index("Parameter")["Coefficient"]

# ==================================================
# Výběr typu zápasu
# ==================================================
game_type_ui = st.radio(
    "Typ zápasu",
    ["Regular Season", "Play-off"],
    horizontal=True
)

game_type_excel = (
    "Regular Season" if game_type_ui == "Regular Season" else "Play-off"
)

#params = (
#    params_raw[params_raw["Game_Type"] == game_type_excel]
#    .assign(Parameter=lambda x: x["Parameter"].astype(str).str.strip())
#    .set_index("Parameter")["Coefficient"]
#)

params = params_raw.set_index("Parameter")["Coefficient"]

# ==================================================
# Výběr týmů
# ==================================================
teams_rs = teams[teams["Game Type"] == "RS"]

team_list = sorted(teams_rs["Team"].unique())

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Domácí tým", team_list)

with col2:
    away_team = st.selectbox(
        "Hosté",
        team_list,
        index=1 if len(team_list) > 1 else 0
    )

def team_strength(team):
    value = teams_rs.loc[
        teams_rs["Team"] == team, "Team_Strength"
    ]

    if value.empty:
        return 0.0  # bezpečnostní pojistka

    return float(value.iloc[0])

team_strength_diff = (
    team_strength(home_team) - team_strength(away_team)
)

# ==================================================
# Vstupy modelu (What‑if)
# ==================================================
st.subheader("Modelové vstupy")

c1, c2, c3, c4 = st.columns(4)

with c1:
    xg_diff = st.slider("xG Diff", -4.0, 4.0, 0.0, 0.05)

with c2:
    pp_diff = st.slider("PP Diff", -1.5, 1.5, 0.0, 0.01)

with c3:
    goalie_rating = st.slider("Goalie rating", -0.50, 0.50, 0.0, 0.01)

with c4:
    home = st.radio("Home", [1, 0], horizontal=True)

# ==================================================
# Model
# ==================================================
def get_param(name, default=0.0):
    if name not in params.index:
        st.warning(f"Chybí parametr: {name}")
        return default
    return float(params[name])

def logistic(x):
    return 1 / (1 + np.exp(-x))

linear_score = (
    get_param("Intercept")
    + home * get_param("Home")
    + xg_diff * get_param("xG_Diff")
    + pp_diff * get_param("PP_Diff")
    + goalie_rating * get_param("Goalie")
    + team_strength_diff * get_param("TeamStrength")
)

p_win = logistic(linear_score)

# ==================================================
# Výstup
# ==================================================
st.divider()
st.subheader("📊 Výsledek predikce")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Lineární skóre", f"{linear_score:.3f}")

with c2:
    st.metric("Výhra domácích", f"{p_win*100:.1f} %")

with c3:
    st.metric("Výhra hostů", f"{(1-p_win)*100:.1f} %")

# ==================================================
# Debug & transparentnost
# ==================================================
with st.expander("🧠 Detail výpočtu"):
    st.write("Koeficienty:")
    st.dataframe(params)

    st.write("Vstupy:")
    st.json({
        "Home": home,
        "xG_Diff": xg_diff,
        "PP_Diff": pp_diff,
        "Goalie_rating": goalie_rating,
        "Team_Strength_Diff": team_strength_diff,
    })