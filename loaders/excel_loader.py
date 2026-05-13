import pandas as pd

def load_excel(path):
    sheets = pd.read_excel(
        path,
        sheet_name=None
    )
    return sheets