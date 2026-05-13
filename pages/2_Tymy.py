import streamlit as st
import pandas as pd

st.set_page_config(page_title="Tymy", layout="wide")
st.title("🏒 Týmy – síla & výkonnost")

# --------------------------------------------------
# Načtení dat
# --------------------------------------------------
@st.cache_data
def load_data():
    xls = pd.ExcelFile("data/HOCKEY_LOGIC_PREDICTIONS.xlsx")
    return pd.read_excel(xls, "CALC_TEAMS_SEASON")

teams = load_data()

# --------------------------------------------------
# Filtry
# --------------------------------------------------
with st.sidebar:
    st.header("🎛️ Filtry")
    season = st.selectbox("Sezóna", sorted(teams["Season"].unique()))
    game_type = st.radio("Typ zápasu", ["RS", "PO"], horizontal=True)

teams = teams[
    (teams["Season"] == season) &
    (teams["Game Type"] == game_type)
]

# --------------------------------------------------
# Přehled týmů
# --------------------------------------------------
st.subheader("📈 Team Strength")

teams_sorted = teams.sort_values("Team_Strength", ascending=False)

st.dataframe(
    teams_sorted[
        [
            "Team",
            "Games",
            "avg_xG_Diff",
            "avg_Shots_Diff",
            "avg_PP_Rate",
            "avg_PK_Rate",
            "Team_Strength"
        ]
    ],
    use_container_width=True
)

st.bar_chart(
    teams_sorted.set_index("Team")["Team_Strength"]
)

# --------------------------------------------------
# Porovnání týmů
# --------------------------------------------------
st.subheader("⚔️ Porovnání týmů")

team_list = teams_sorted["Team"].tolist()

c1, c2 = st.columns(2)

with c1:
    team_a = st.selectbox("Tým A", team_list)

with c2:
    team_b = st.selectbox("Tým B", team_list, index=1)

ta = teams_sorted[teams_sorted["Team"] == team_a]
tb = teams_sorted[teams_sorted["Team"] == team_b]

comparison = pd.DataFrame({
    team_a: ta.iloc[0][
        ["avg_xG_Diff", "avg_Shots_Diff", "avg_PP_Rate", "avg_PK_Rate", "Team_Strength"]
    ],
    team_b: tb.iloc[0][
        ["avg_xG_Diff", "avg_Shots_Diff", "avg_PP_Rate", "avg_PK_Rate", "Team_Strength"]
    ],
})

st.dataframe(comparison)

# --------------------------------------------------
# Rychlé insighty
# --------------------------------------------------
st.subheader("🧠 Rychlé závěry")

diff = ta["Team_Strength"].values[0] - tb["Team_Strength"].values[0]

if diff > 0:
    st.success(f"{team_a} má vyšší Team Strength o {diff:.2f}")
else:
    st.warning(f"{team_b} má vyšší Team Strength o {abs(diff):.2f}")
``