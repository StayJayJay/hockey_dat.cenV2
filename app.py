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

        """
    )

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
    Vítej v analytické aplikaci **ELH**

    ---
    ### Co tady najdeš:
    **Statistiky**

    **Týmy**
    - Team Strength
    - PP / PK
    - porovnání

    **Predikce**
    - What‑if analýza
    - transparentní výpočet P(Win)

    ---
    """
)

# ==================================================
# Přehled dostupných datových sheetů (debug / info)
# ==================================================
# with st.expander("📂 Načtené listy z Excelu"):
#    st.write(list(sheets.keys()))