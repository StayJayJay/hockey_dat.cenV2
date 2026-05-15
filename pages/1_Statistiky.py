# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Statistiky týmů", layout="wide")
st.title("📊 Statistiky týmů")

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
# FEATURE ENGINEERING (jen základní)
# ==================================================
df = df.sort_values("Date")

df["Team_form"] = (
    df.groupby("Team")["Win"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# ==================================================
# UI – výběr týmu
# ==================================================
team = st.selectbox("Vyber tým", sorted(df["Team"].unique()))

team_df = df[df["Team"] == team].copy()

if team_df.empty:
    st.warning("Žádná data pro tento tým")
    st.stop()

# ==================================================
# ZÁKLADNÍ METRIKY
# ==================================================
st.subheader("📊 Přehled")

win_rate = team_df["Win"].mean()
avg_xg = team_df["xG_Diff_adj"].mean()
avg_pp = team_df["PP_Diff"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("✅ Win rate", f"{win_rate*100:.1f}%")
col2.metric("⚡ Avg xG diff", f"{avg_xg:.2f}")
col3.metric("🔥 PP diff", f"{avg_pp:.2f}")

# ==================================================
# ⚔️ POROVNÁNÍ TEAM vs OPPONENT
# ==================================================
st.subheader("⚔️ Porovnání týmů")

opponent = st.selectbox("Vyber soupeře", sorted(df["Team"].unique()))

opp_df = df[df["Team"] == opponent]

if not opp_df.empty:

    col1, col2 = st.columns(2)

    col1.write(f"### {team}")
    col1.write(f"Forma: {team_df['Team_form'].iloc[-1]:.2f}")
    col1.write(f"Win rate: {team_df['Win'].mean():.2%}")

    col2.write(f"### {opponent}")
    col2.write(f"Forma: {opp_df['Team_form'].iloc[-1]:.2f}")
    col2.write(f"Win rate: {opp_df['Win'].mean():.2%}")


# ==================================================
# FORMA V ČASE
# ==================================================
st.subheader("📈 Forma týmu (poslední zápasy)")

st.line_chart(team_df.set_index("Date")["Team_form"])

# ==================================================
# 📉 TREND FORMY
# ==================================================
st.subheader("📉 Trend formy")

recent_form = team_df["Win"].tail(5).mean()
season_form = team_df["Win"].mean()

if recent_form > season_form:
    st.success("📈 Tým se zlepšuje")
else:
    st.warning("📉 Tým se zhoršuje")


# ==================================================
# POSLEDNÍ ZÁPASY
# ==================================================
st.subheader("📅 Poslední zápasy")

st.dataframe(
    team_df.tail(10)[
        ["Date", "Opponent", "Win", "Team_form"]
    ]
)

# ==================================================
# HOME vs AWAY
# ==================================================
st.subheader("🏠 Home vs Away")

home_games = team_df[team_df["Home"] == 1]["Win"].mean()
away_games = team_df[team_df["Home"] == 0]["Win"].mean()

col1, col2 = st.columns(2)
col1.metric("🏠 Home win rate", f"{home_games*100:.1f}%")
col2.metric("✈️ Away win rate", f"{away_games*100:.1f}%")

# ==================================================
# DISTRIBUCE VÝSLEDKŮ
# ==================================================
st.subheader("📊 Distribuce výsledků")

st.bar_chart(team_df["Win"])

# ==================================================
# SILNÁ / SLABÁ STRÁNKA
# ==================================================
st.subheader("🧠 Insight")

latest_form = team_df["Team_form"].iloc[-1]

if latest_form > 0.7:
    st.success("🔥 Tým je ve výborné formě")
elif latest_form > 0.55:
    st.info("✅ Tým je ve solidní formě")
else:
    st.warning("⚠️ Tým je ve slabší formě")

# ==================================================
# 🏆 TOP TÝMY
# ==================================================
st.subheader("🏆 Top týmy")

leaderboard = df.groupby("Team")["Win"].mean().sort_values(ascending=False)

st.dataframe(
    leaderboard.reset_index().rename(columns={"Win": "Win rate"})
)
