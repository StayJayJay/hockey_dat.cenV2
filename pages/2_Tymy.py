# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Týmy", layout="wide")
st.title("Týmy – Team Strength & výkonnost")

# --------------------------------------------------
# Načtení dat (stejný pattern jako app.py)
# --------------------------------------------------
@st.cache_data
def load_data():
    base_path = Path(__file__).resolve().parent.parent
    file_path = base_path / "data" / "HOCKEY_LOGIC_PREDICTIONS.xlsx"

    if not file_path.exists():
        st.error(f"Excel nenalezen: {file_path}")
        st.stop()

    return pd.read_excel(file_path, sheet_name="CALC_TEAMS_SEASON")

teams = load_data()

# --------------------------------------------------
# Sidebar filtry
# --------------------------------------------------
with st.sidebar:
    st.header("Filtry")

    season = st.selectbox(
        "Sezóna",
        sorted(teams["Season"].unique())
    )

    game_type = st.radio(
        "Typ zápasu",
        ["RS", "PO"],
        horizontal=True
    )

# --------------------------------------------------
# Filtrování dat
# --------------------------------------------------
teams = teams[
    (teams["Season"] == season) &
    (teams["Game Type"] == game_type)
].copy()

if teams.empty:
    st.warning("Žádná data pro zvolenou kombinaci.")
    st.stop()

# --------------------------------------------------
# Team Strength – tabulka
# --------------------------------------------------
st.subheader("📈 Síla týmů (Team Strength)")

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
            "Team_Strength",
        ]
    ],
    use_container_width=True
)

st.bar_chart(
    teams_sorted.set_index("Team")["Team_Strength"]
)

# --------------------------------------------------
# Porovnání dvou týmů
# --------------------------------------------------
st.subheader("⚔️ Porovnání týmů")

team_list = teams_sorted["Team"].tolist()

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Tým A", team_list)

with col2:
    team_b = st.selectbox(
        "Tým B",
        team_list,
        index=1 if len(team_list) > 1 else 0
    )

ta = teams_sorted[teams_sorted["Team"] == team_a].iloc[0]
tb = teams_sorted[teams_sorted["Team"] == team_b].iloc[0]

comparison = pd.DataFrame(
    {
        team_a: [
            ta["avg_xG_Diff"],
            ta["avg_Shots_Diff"],
            ta["avg_PP_Rate"],
            ta["avg_PK_Rate"],
            ta["Team_Strength"],
        ],
        team_b: [
            tb["avg_xG_Diff"],
            tb["avg_Shots_Diff"],
            tb["avg_PP_Rate"],
            tb["avg_PK_Rate"],
            tb["Team_Strength"],
        ],
    },
    index=[
        "avg xG Diff",
        "avg Shots Diff",
        "avg PP Rate",
        "avg PK Rate",
        "Team Strength",
    ],
)

st.dataframe(comparison, use_container_width=True)

# --------------------------------------------------
# Shrnutí
# --------------------------------------------------
diff_TeamStrength = ta["Team_Strength"] - tb["Team_Strength"]

if diff_TeamStrength > 0:
    st.success(f"{team_a} má vyšší Team Strength o {diff_TeamStrength:.2f}")
elif diff_TeamStrength < 0:
    st.warning(f"{team_b} má vyšší Team Strength o {abs(diff_TeamStrength):.2f}")
else:
    st.info("Týmy mají stejnou Team Strength")

# ---------------------------------------------------

diff_xG = ta["avg_xG_Diff"] - tb["avg_xG_Diff"]

if diff_xG > 0:
    st.success(f"{team_a} má vyšší průměrný xG diff o {diff_xG:.2f}")
elif diff_xG < 0:
    st.warning(f"{team_b} má vyšší průměrný xG diff o {abs(diff_xG):.2f}")
else:
    st.info("Týmy mají stejný průměrný rozdíl xG")

# -----------------------------------------------------

diff_PP_rate = ta["avg_PP_Rate"] - tb["avg_PP_Rate"]

if diff_PP_rate > 0:
    st.success(f"{team_a} má vyšší průměrné využití o {diff_PP_rate:.2f}")
elif diff_PP_rate < 0:
    st.warning(f"{team_b} má vyšší průměrné využití o {abs(diff_PP_rate):.2f}")
else:
    st.info("Týmy mají stejné průměrné využití PP")