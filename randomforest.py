import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
config = f"{directory}\\config\\"
storage = f"{directory}\\data\\"

commodities = ["LC1", "CO1", "CL1", "CT1", "NG1", "HG1", "W1", "GC1", "S1"]
commodity = "NG1"

data = pd.read_pickle(f"{storage}{commodity}.pkl")

predictors = [f"{commodity}_MA_5", f"{commodity}_MA_10", f"{commodity}_MA_15",
              f"{commodity}_SD_5", f"{commodity}_HL", f"{commodity}_OC"]

predictors = [f"{commodity}_PX_LAST", f"{commodity}_PX_LOW", f"{commodity}_PX_HIGH", f"{commodity}_PX_OPEN",
              f"{commodity}_PX_VOLUME", f"{commodity}_OPEN_INT"]

train = data.iloc[:-250]
test = data.iloc[-250:]

model = RandomForestClassifier(n_estimators=1000, min_samples_split=100, random_state=13)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

predictions = backtest(data, model, predictors)

print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))
print(predictions["Target"].value_counts() / predictions.shape[0])

print()