import os
from sklearn.linear_model import LinearRegression
import pandas as pd


script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
storage = os.path.join(directory, 'data')
commodities = ['LC1', 'CO1', 'HG1', 'GC1', 'S1']
split_date = '2023-05-01'
gap_days = 40


for commodity in commodities:
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    data.dropna(inplace=True)
    data.index = pd.to_datetime(data.index)

    predictors = [f'{commodity}_MACD', f'{commodity}_RSI']

    test_start_date = pd.to_datetime(split_date)
    gap_start_date = test_start_date - pd.DateOffset(days=gap_days)
    train = data[data.index < gap_start_date]
    test = data[data.index >= test_start_date]
    gap = data[(data.index >= gap_start_date) & (data.index < test_start_date)]

    X_train = train[predictors]
    y_train = train[f'{commodity}_Price_Tomorrow']
    X_test = test[predictors]
    y_test = test[f'{commodity}_Price_Tomorrow']

    model_lin = LinearRegression()
    model_lin.fit(X_train, y_train)
    model_lin.score(X_train, y_train)
    regline = model_lin.predict(X_train)
    predictions_lin = model_lin.predict(X_test)

    results_lin = pd.DataFrame({
        'Price': y_test,
        'Pred_Price': predictions_lin}, index=test.index)
    results_lin.to_pickle(os.path.join(storage, 'predictions', 'linear', f'{commodity}_linear_pred.pkl'))