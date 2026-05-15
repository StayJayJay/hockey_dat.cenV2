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
# TRAIN / TEST SPLIT
# ==================================================
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(
    df,
    test_size=0.3,
    random_state=42
)

st.write("Train size:", len(df_train))
st.write("Test size:", len(df_test))

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
# AUTO-TUNING SCALE (rychlá grid search)
# ==================================================
def tune_scaling(df_sample):

    best_score = 999
    best_params = None

    # testované kombinace
    xg_scales = [0.1, 0.15, 0.2]
    pp_scales = [2, 3, 4]
    goalie_scales = [15, 25, 35]

    for xg_s in xg_scales:
        for pp_s in pp_scales:
            for g_s in goalie_scales:

                preds = []

                for _, row in df_sample.iterrows():

                    xg_raw = row.get("xG_Diff_adj")
                    shots = safe_val(row.get("Shots_Diff"))

                    if pd.notna(xg_raw) and abs(xg_raw) > 0:
                        quality = xg_raw
                        weight = 1.0
                    else:
                        quality = shots * 0.1
                        weight = 0.6

                    score = (
                        get_param("Intercept")
                        + safe_val(row.get("Home")) * get_param("Home")
                        + (quality * xg_s) * get_param("xG_Diff") * weight
                        + safe_val(row.get("PP_Diff")) * pp_s * get_param("PP_Diff")
                        + safe_val(row.get("Goalie_rating")) * g_s * get_param("Goalie")
                        + safe_val(row.get("Team_strength")) * get_param("TeamStrength")
                    )

                    p = logistic(score)
                    p = 0.5 + (p - 0.5) * 0.6

                    preds.append(p)

                # Brier score
                brier = np.mean((np.array(preds) - df_sample["Win"])**2)

                if brier < best_score:
                    best_score = brier
                    best_params = (xg_s, pp_s, g_s)

    return best_params

best_xg, best_pp, best_goalie = tune_scaling(df.sample(200))

st.write("Best scaling:")
st.write("xG:", best_xg, "PP:", best_pp, "Goalie:", best_goalie)

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
    quality_scaled = quality * 0.2

    # PP
    pp_scaled = pp * 4

    # Goalie
    goalie_scaled = goalie * 15

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

df_train["P_pred"] = df_train.apply(predict, axis=1)
df_test["P_pred"] = df_test.apply(predict, axis=1)


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
df_test["Predicted_Class"] = (df_test["P_pred"] > 0.5).astype(int)
accuracy = (df_test["Predicted_Class"] == df_test["Win"]).mean()

# Brier score
df_clean = df_test.dropna(subset=["P_pred", "Win"])
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
    df_test["P_pred"]
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

#================================================================================================================
#================================================================================================================
#================================================================================================================
#================================================================================================================

#Seřazení podle datumu
df = df.sort_values(by="Date")

# ==================================================
# TEAM FORM (last 5 games)
# ==================================================
df["Team_form"] = (
    df.groupby("Team")["Win"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
# vytvoř pomocnou tabulku
team_form_lookup = df[["Team", "Date", "Team_form"]].copy()
team_form_lookup.columns = ["Opponent", "Date", "Opponent_form"]

# merge zpátky podle soupeře a času
df = df.merge(
    team_form_lookup,
    on=["Opponent", "Date"],
    how="left"
)



# ==================================================
# ML MODEL (Logistic Regression)
# ==================================================
try:
    from sklearn.linear_model import LogisticRegression

    st.subheader("ML model")

    # --- feature engineering ---
    df_ml = df.copy()

    df_ml["quality"] = df_ml["xG_Diff_adj"]
    mask = df_ml["quality"].isna() | (df_ml["quality"] == 0)
    df_ml.loc[mask, "quality"] = df_ml["Shots_Diff"] * 0.1

    features = ["Home","PP_Diff","Goalie_rating","Team_strength","quality","Team_form","Opponent_form"]

    # split stejné jako výš
    df_ml_shuffled = df_ml.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(df_ml_shuffled) * 0.7)

    df_ml_train = df_ml_shuffled.iloc[:split_index]
    df_ml_test = df_ml_shuffled.iloc[split_index:]

    X_train = df_ml_train[features]
    y_train = df_ml_train["Win"]

    X_test = df_ml_test[features]
    y_test = df_ml_test["Win"]

    # --- model ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # --- predikce ---
    df_ml_test["P_pred_ml"] = model.predict_proba(X_test)[:, 1]
    df_ml_test["Predicted_Class_ml"] = (df_ml_test["P_pred_ml"] > 0.5).astype(int)

    # --- metriky ---
    accuracy_ml = (df_ml_test["Predicted_Class_ml"] == y_test).mean()
    brier_ml = np.mean((df_ml_test["P_pred_ml"] - y_test) ** 2)

    # --- output ---
    col1, col2 = st.columns(2)

    with col1:
        st.metric("ML Accuracy", f"{accuracy_ml*100:.1f} %")

    with col2:
        st.metric("ML Brier", f"{brier_ml:.4f}")

except ImportError:
    st.warning(" scikit-learn není nainstalovaný (přidej do requirements.txt)")
