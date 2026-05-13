import streamlit as st
import pandas as pd

st.set_page_config(page_title="Statistiky", layout="wide")
st.title("📊 Statistiky zápasů – ELH")

# --------------------------------------------------
# Načtení dat
# --------------------------------------------------
@st.cache_data
def load_data():
    xls = pd.ExcelFile("data/HOCKEY_LOGIC_PREDICTIONS.xlsx")
    return pd.read_excel(xls, "CALC_GAMES_BASE")

df = load_data()

# --------------------------------------------------
# Filtry
# --------------------------------------------------
with st.sidebar:
    st.header("🎛️ Filtry")
    season = st.selectbox("Sezóna", sorted(df["Season"].unique()))
    game_type = st.multiselect(
        "Typ zápasu",
        df["Game_Type"].unique(),
        default=df["Game_Type"].unique()
    )

df = df[
    (df["Season"] == season) &
    (df["Game_Type"].isin(game_type))
]

# --------------------------------------------------
# Rychlé metriky
# --------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Zápasy", len(df))

with c2:
    st.metric("Góly / zápas", f"{df['Goals_For'].mean():.2f}")

with c3:
    st.metric("xG / zápas", f"{df['xG_For'].mean():.2f}")

with c4:
    st.metric("PP %", f"{df['PP_Eff'].mean()*100:.1f} %")

# --------------------------------------------------
# Přesilovky – týmově
# --------------------------------------------------
st.subheader("⚡ Přesilovky – týmové")

pp = (
    df.groupby("Team")[["PP_O", "PP_G"]]
    .sum()
)

pp["PP_%"] = pp["PP_G"] / pp["PP_O"]
pp = pp.sort_values("PP_%", ascending=False)

st.dataframe(
    pp.style.format({"PP_%": "{:.1%}"}),
    use_container_width=True
)

st.bar_chart(pp["PP_%"])

# --------------------------------------------------
# Detail zápasů
# --------------------------------------------------
st.subheader("📋 Detail zápasů")

st.dataframe(
    df.sort_values("Date", ascending=False),
    use_container_width=True
)