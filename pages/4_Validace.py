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

# ==================================================
# Scaling helper
# ==================================================
def safe_val(x):
    if pd.isna(x):
        return 0
    return x


# ==================================================
# Model s xG / Shots fallback + scaling
# ==================================================
def predict(row):
    # --- RAW vstupy ---
    home = safe_val(row.get("Home"))
    xg_raw = row.get("xG_Diff_adj")
    shots = safe_val(row.get("Shots_Diff"))  # fallback
    pp = safe_val(row.get("PP_Diff"))
    goalie = safe_val(row.get("Goalie_rating"))
    strength = safe_val(row.get("Team_strength"))

    # ==================================================
    # xG FEATURE LOGIC (KLÍČOVÉ)
    # ==================================================

    # pokud xG existuje → použij xG
    if pd.notna(xg_raw) and abs(xg_raw) > 0:
        quality = xg_raw
        quality_weight = 1.0
    else:
        # fallback → použij střely
        quality = shots * 0.1
        quality_weight = 0.6

    # ==================================================
    # SCALING
    # ==================================================

    # kvalita (xG nebo Shots)
    quality_scaled = quality * 0.15

    # PP
    pp_scaled = pp * 3

    # Goalie
    goalie_scaled = goalie * 25

    # Team strength
    strength_scaled = strength

    # ==================================================
    # Výpočet score
    # ==================================================
    score = (
        get_param("Intercept")
        + home * get_param("Home")
        + quality_scaled * get_param("xG_Diff") * quality_weight
        + pp_scaled * get_param("PP_Diff")
        + goalie_scaled * get_param("Goalie")
        + strength_scaled * get_param("TeamStrength")
    )

    # ==================================================
    # logistická transformace
    # ==================================================
    p = logistic(score)

    # ==================================================
    # CALIBRATION SHRINK (fix Brier)
    # ==================================================
    p = 0.5 + (p - 0.5) * 0.6

    if p > 0.65:
        p = 0.65 + (p - 0.65) * 0.6

    return p

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

#st.write("Použité sloupce:", existing_cols)
#st.write("Počet řádků po čištění:", len(df))
#st.write("Min prob:", df["P_pred"].min())
#st.write("Max prob:", df["P_pred"].max())
#st.write("Podíl xG vs Shots:")
#st.write((df["xG_Diff_adj"] != 0).mean())

# ==================================================
# METRIKY
# ==================================================

# Accuracy
df["Predicted_Class"] = (df["P_pred"] > 0.5).astype(int)
accuracy = (df["Predicted_Class"] == df["Win"]).mean()

# Brier score
df_clean = df.dropna(subset=["P_pred", "Win"])
brier = np.mean((df_clean["P_pred"] - df_clean["Win"]) ** 2)


# ==================================================
# KALIBRACE MODELU
# ==================================================
st.subheader("📈 Kalibrace modelu")

# vytvoření binů
df_clean["bin"] = pd.cut(df_clean["P_pred"], bins=10)

calibration = df_clean.groupby("bin").agg(
    avg_pred=("P_pred", "mean"),
    actual_win_rate=("Win", "mean"),
    count=("Win", "count")
).reset_index()

# filtr na dost dat
calibration = calibration[calibration["count"] > 10]

st.line_chart(
    calibration.set_index("avg_pred")[["actual_win_rate"]]
)

st.write(calibration)

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

#st.subheader("🔍 Debug – sloupce v datasetu")
#st.write(df.columns.tolist())

#st.subheader("🔍 Počet řádků před filtrem")
#st.write(len(df))