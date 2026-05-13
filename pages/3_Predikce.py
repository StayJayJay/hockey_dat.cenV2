import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Predikce zápasu", layout="wide")
st.title("🔮 Predikce zápasu – ELH")

# --------------------------------------------------
# Načtení dat
# --------------------------------------------------
@st.cache_data
def load_data():
    xls = pd.ExcelFile("data/HOCKEY_LOGIC_PREDICTIONS.xlsx")
    return {
        "PARAMETRY": pd.read_excel(xls, "PARAMETRY"),
        "PREDIKCE": pd.read_excel(xls, "PREDIKCE_ZAPASU"),
        "TEAMS": pd.read_excel(xls, "CALC_TEAMS_SEASON"),
    }

data = load_data()

params_raw = data["PARAMETRY"]
teams = data["TEAMS"]

# --------------------------------------------------
# Výběr typu zápasu
# --------------------------------------------------
game_type = st.radio(
    "Typ zápasu",
    ["Regular Season", "Play-off"],
    horizontal=True
)

params = params_raw[
    (params_raw["Game Type"] == ("RS" if game_type == "Regular Season" else "PO"))
].set_index("Parameter")["Coefficient"]

# --------------------------------------------------
# Výběr týmů
# --------------------------------------------------
teams_rs = teams[teams["Game Type"] == "RS"]

team_list = sorted(teams_rs["Team"].unique())

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("🏠 Domácí tým", team_list)

with col2:
    away_team = st.selectbox("✈️ Hosté", team_list, index=1)

# --------------------------------------------------
# Načtení team strength
# --------------------------------------------------
def get_team_strength(team):
    return float(
        teams_rs.loc[teams_rs["Team"] == team, "Team_Strength"]
    )

home_strength = get_team_strength(home_team)
away_strength = get_team_strength(away_team)

team_strength_diff = home_strength - away_strength

# --------------------------------------------------
# Manuální vstupy (What-if mód)
# --------------------------------------------------
st.subheader("⚙️ Modelové vstupy")

c1, c2, c3, c4 = st.columns(4)

with c1:
    xg_diff = st.slider("xG Diff", -3.0, 3.0, 0.0, 0.05)

with c2:
    pp_diff = st.slider("PP Diff", -0.5, 0.5, 0.0, 0.01)

with c3:
    goalie_rating = st.slider("Goalie rating", -0.05, 0.05, 0.0, 0.001)

with c4:
    home = st.radio("Home", [1, 0], horizontal=True)

# --------------------------------------------------
# Výpočet predikce (1:1 podle Excelu)
# --------------------------------------------------
def logistic(x):
    return 1 / (1 + np.exp(-x))

linear_score = (
    params["Intercept"]
    + home * params["Home"]
    + xg_diff * params["xG_Diff"]
    + pp_diff * params["PP_Diff"]
    + goalie_rating * params["Goalie"]
    + team_strength_diff * params["TeamStrength"]
)

p_win = logistic(linear_score)

# --------------------------------------------------
# Výstup
# --------------------------------------------------
st.divider()
st.subheader("📊 Výsledek predikce")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Lineární skóre", f"{linear_score:.3f}")

with c2:
    st.metric("Pravděpodobnost výhry domácích", f"{p_win*100:.1f} %")

with c3:
    st.metric("Pravděpodobnost výhry hostů", f"{(1-p_win)*100:.1f} %")

# --------------------------------------------------
# Debug / transparentnost
# --------------------------------------------------
with st.expander("🧠 Detail výpočtu"):
    st.write("Použité koeficienty:")
    st.dataframe(params)

    st.write("Použité hodnoty:")
    st.json({
        "Home": home,
        "xG_Diff": xg_diff,
        "PP_Diff": pp_diff,
        "Goalie_rating": goalie_rating,
        "Team_Strength_Diff": team_strength_diff,
    })