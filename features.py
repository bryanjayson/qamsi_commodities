import numpy as np
import pandas as pd
import os
import sklearn as sk

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
config = f"{directory}\\config\\"
storage = f"{directory}\\data\\"

data = pd.read_excel(f"{directory}/data/commodity_data.xlsx")
data["Dates"] = pd.to_datetime(["Dates"])

scaler = sk.MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

print()