import numpy as np

def logistic(x):
    return 1 / (1 + np.exp(-x))


def predict(row, params):
    score = (
        params["Intercept"]
        + row["Home"] * params["Home"]
        + row["xG_Diff"] * params["xG_Diff"]
        + row["PP_Diff"] * params["PP_Diff"]
        + row["Goalie_rating"] * params["Goalie"]
        + row["Team_Strength"] * params["TeamStrength"]
    )

    return logistic(score)