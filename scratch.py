import pandas as pd
import os

current_dir = os.path.dirname(__file__)
data_storage = os.path.join('data')
prediction_storage = os.path.join('data', 'predictions')

viewer = pd.read_pickle(os.path.join(current_dir, prediction_storage, 'lstm', 'LC1_lstm_pred.pkl'))
viewer_2 = pd.read_pickle(os.path.join(current_dir, prediction_storage, 'rf', 'LC1_rf_pred.pkl'))
viewer_3 = pd.read_pickle(os.path.join(current_dir, prediction_storage, 'linear', 'LC1_linear_pred.pkl'))
viewer_4 = pd.read_pickle(os.path.join(current_dir, data_storage, 'raw', 'LC1.pkl'))

print()