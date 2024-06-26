import pandas as pd
import os

current_dir = os.path.dirname(__file__)
data_storage = os.path.join('data')
prediction_storage = os.path.join('data', 'predictions')

viewer = pd.read_pickle(os.path.join(current_dir, prediction_storage, 'lstm', 'NG1_lstm_pred.pkl'))
viewer.to_clipboard()
viewer_2 = pd.read_pickle(os.path.join(current_dir, prediction_storage, 'rf', 'NG1_rf_pred.pkl'))
viewer_2.to_clipboard()
viewer_3 = pd.read_pickle(os.path.join(current_dir, data_storage, 'raw', 'CO1.pkl'))

print()