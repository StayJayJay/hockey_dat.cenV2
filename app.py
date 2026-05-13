import streamlit as st
import pandas as pd

# ==================================================
# Základní nastavení aplikace
# ==================================================
st.set_page_config(
    page_title="ELH – Statistiky & Predikce",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# Sidebar – info
# ==================================================
with st.sidebar:
    st.title("🏒 ELH Analytics")
    st.markdown(
        """
        **Statistická a predikční aplikace ELH**

        - týmové statistiky  
        - přesilovky / oslabení  
        - síla týmů  
        - predikce zápasů  

        ---
        **Model**
        - lineární (logistic)
        - koeficienty z Excelu
        """
    )

    st.markdown("📁 **Zdroj dat:**")
    st.code("HOCKEY_LOGIC_PREDICTIONS.xlsx", language="text")

# ==================================================
# Cacheované načtení Excelu
# ==================================================
@st.cache_data
def load_excel():
    file_path = "data/HOCKEY_LOGIC_PREDICTIONS.xlsx"
    sheets = pd.read_excel(file_path, sheet_name=None)
    return sheets

# ==================================================
# Data k dispozici pro všechny stránky
# ==================================================
sheets = load_excel()

# ==================================================
# Home page (úvod)
# ==================================================
st.title("🏒 ELH – Statistiky & Predikce")

st.markdown(
    """
    Vítej v analytické aplikaci **ELH**, která je postavená **přímo nad tvým
    modelem v Excelu** – bez zjednodušování a bez ztráty logiky.

    ---
    ### 🧭 Co tady najdeš:
    **📊 Statistiky**
    - zápasová data
    - přesilovky, oslabení
    - střely, xG, góly

    **🏒 Týmy**
    - Team Strength
    - PP / PK
    - forma a srovnání

    **🔮 Predikce**
    - přepínání RS / PO
    - What‑if analýza
    - transparentní výpočet P(Win)

    ---
    """
)

# ==================================================
# Přehled dostupných datových sheetů (debug / info)
# ==================================================
with st.expander("📂 Načtené listy z Excelu"):
    st.write(list(sheets.keys()))

st.markdown(
    """
    ---
    👉 **Použij navigaci vlevo** pro přechod na konkrétní část aplikace.
    """
)
