# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Validace modelu", layout="wide")
st.title("📊 Validace predikčního modelu")

# ==================================================
# Načtení dat
# ==================================================
@st.cache_data
def load_data():
    base_path = Path(__file__).resolve().parent.parent
    file_path = base_path / "data" / "HOCKEY_LOGIC_PREDICTIONS.xlsx"

    model_input = pd.read_excel(
        file_path,
        sheet_name="MODEL_INPUT"
    )

    params_raw = pd.read_excel(
        file_path,
        sheet_name="PARAMETRY",
        header=1
    )

    return model_input, params_raw


df, params_raw = load_data()

# ==================================================
# Parametry
# ==================================================
params_raw = params_raw.iloc[:, :4]
params_raw.columns = ["Parameter", "Coefficient", "Source", "Note"]
params_raw = params_raw.dropna(subset=["Parameter", "Coefficient"])
params_raw["Parameter"] = params_raw["Parameter"].astype(str).str.strip()

params = params_raw.set_index("Parameter")["Coefficient"]

# bezpečný getter
def get_param(name, default=0.0):
    if name not in params.index:
        return default
    return float(params[name])

# ==================================================
# Model
# ==================================================
def logistic(x):
    return 1 / (1 + np.exp(-x))

def predict(row):
    score = (
        get_param("Intercept")
        + row.get("Home", 0) * get_param("Home")
        + row.get("xG_Diff_adj", 0) * get_param("xG_Diff")
        + row.get("PP_Diff", 0) * get_param("PP_Diff")
        + row.get("Goalie_rating", 0) * get_param("Goalie")
        + row.get("Team_strength", 0) * get_param("TeamStrength")
    )
    return logistic(score)

# ==================================================
# Predikce na historických datech
# ==================================================

df["P_pred"] = df.apply(predict, axis=1)

# ==================================================
# ČIŠTĚNÍ DAT (správné)
# ==================================================
required_cols = [
    "Home",
    "xG_Diff_adj",
    "PP_Diff",
    "Goalie_rating",
    "Team_strength",
    "Win"
]

# nech jen existující sloupce
existing_cols = [col for col in required_cols if col in df.columns]

# 🔑 místo dropna → doplníme chybějící hodnoty
df[existing_cols] = df[existing_cols].fillna(0)

st.write("Použité sloupce:", existing_cols)
st.write("Počet řádků po čištění:", len(df))

# ==================================================
# METRIKY
# ==================================================

# Accuracy
df["Predicted_Class"] = (df["P_pred"] > 0.5).astype(int)
accuracy = (df["Predicted_Class"] == df["Win"]).mean()

# Brier score
brier = np.mean((df["P_pred"] - df["Win"]) ** 2)

# ==================================================
# OUTPUT
# ==================================================
st.subheader("📈 Výsledky modelu")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy*100:.1f} %")

with col2:
    st.metric("Brier score", f"{brier:.4f}")

# ==================================================
# DISTRIBUCE
# ==================================================
st.subheader("📊 Distribuce predikcí")

st.bar_chart(
    df["P_pred"]
)

if df.empty:
    st.error("❌ Žádná data pro validaci – zkontroluj MODEL_INPUT sheet")
    st.stop()

# ==================================================
# DEBUG DATA
# ==================================================
with st.expander("📋 Data sample"):
    st.dataframe(df.head())

st.subheader("🔍 Debug – sloupce v datasetu")
st.write(df.columns.tolist())

st.subheader("🔍 Počet řádků před filtrem")
st.write(len(df))