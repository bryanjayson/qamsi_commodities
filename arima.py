from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from arch import arch_model
import pandas as pd
import numpy as np
import os

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
config = os.path.join(directory, 'config')
storage = os.path.join(directory, 'data')

commodities = ['LC1', 'CO1', 'CT1', 'NG1', 'HG1', 'W1', 'GC1', 'S1']


def arima_mod(data, t_set_size=0.8, order=(1, 1, 1)):
    data.index = pd.DatetimeIndex(data.index, freq='B')

    data['Differenced'] = data['Tomorrow'].diff().dropna()

    split_point = int(len(data['Differenced']) * t_set_size)

    train = data['Differenced'][:split_point]
    test = data['Differenced'][split_point:]

    model = ARIMA(train, order=order)
    model_fit = model.fit()

    start = len(train)
    end = start + len(test) - 1
    predictions = model_fit.predict(start=start, end=end, typ='levels')

    predictions.index = test.index
    predictions = predictions.cumsum()

    # Adding the last actual price to the cumulative sum predictions to restore original scale
    predictions += data['Tomorrow'].iloc[split_point - 1]

    combined_df = pd.concat([data['Tomorrow'][split_point:], predictions], axis=1)
    combined_df.columns = ['Tomorrow_Actual', 'Predictions_Price']

    return combined_df


def auto_arima_mod(data, commodity, t_set_size=0.8, seasonal_period=5):
    data.index = pd.DatetimeIndex(data.index, freq='B')
    data = data.dropna()

    split_point = int(len(data) * t_set_size)
    train = data['Log_Returns'][:split_point]
    test = data['Log_Returns'][split_point:]

    result = data['Log_Returns'][:split_point]
    p_value = result[1]

    if p_value < 0.05:
        print(f"The time series of returns for {commodity} is stationary.")
    else:
        print(f"The time series of returns for {commodity} is not stationary.")


    model = auto_arima(train, d=1, start_p=1, start_q=1,
                       test='adf',
                       max_p=25, max_q=25,
                       m=seasonal_period,
                       seasonal=True,
                       stepwise=True,
                       suppress_warnings=True,
                       error_action='ignore',
                       trace=True)

    predicted_log_returns = model.predict(n_periods=len(test))
    predicted_log_returns = pd.Series(predicted_log_returns, index=test.index)

    last_train_price = data[f'{commodity}_PX_LAST'].iloc[split_point - 1]
    actual_prices = data[f'{commodity}_PX_LAST'][split_point:]
    predicted_prices = np.exp(predicted_log_returns.cumsum()) * last_train_price

    actual_direction = data['Direction'][split_point:]
    predicted_direction = (predicted_log_returns > 0).astype(int)

    combined_df = pd.concat([actual_prices, predicted_prices, test, predicted_log_returns, actual_direction, predicted_direction], axis=1)
    combined_df.columns = ['Price', 'Pred_Price', 'Log_Ret', 'Pred_Log_Ret', 'Dir', 'Pred_Dir']
    combined_df = combined_df.reset_index()
    combined_df.columns = ['Date', 'Price', 'Pred_Price', 'Log_Ret', 'Pred_Log_Ret', 'Dir', 'Pred_Dir']
    combined_df.set_index('Date', inplace=True)
    combined_df.index = pd.to_datetime(combined_df.index).date

    if combined_df.isna().any().any():
        print(f'Warning: Predictions for {commodity} contain NaN values.')

    return combined_df


def fit_garch(returns, max_p, max_q):
    opt_aic = np.inf
    opt_order = None
    opt_model = None

    p_range = range(1, max_p)
    q_range = range(1, max_q)

    for p in p_range:
        for q in q_range:
            try:
                model = arch_model(returns, vol='GARCH', p=p, q=q)
                model_fit = model.fit(disp='off')
                aic = model_fit.aic

                if aic < opt_aic:
                    opt_aic = aic
                    opt_order = (p, q)
                    opt_model = model_fit

            except Exception as e:
                continue

    return opt_model, opt_order, opt_aic


def auto_garch_mod(data, commodity, max_p, max_q, t_set_size=0.8):
    data.index = pd.DatetimeIndex(data.index, freq='B')
    data = data.dropna()

    split_point = int(len(data) * t_set_size)
    train = data['Log_Returns'][:split_point]
    test = data['Log_Returns'][split_point:]

    adf_result = adfuller(train)
    p_value = adf_result[1]

    if p_value < 0.05:
        print(f"The time series of returns for {commodity} is stationary.")
    else:
        print(f"The time series of returns for {commodity} is not stationary.")

    opt_model, opt_order, opt_aic = fit_garch(train, max_p, max_q)

    forecast = opt_model.forecast(horizon=len(test))
    if 'h.1' not in forecast.mean.columns:
        raise Exception('Forecast does not contain h.1. Check model configuration and forecast horizon.')

    predicted_log_returns = forecast.mean['h.1'].iloc[-len(test):]

    last_train_price = data[f'{commodity}_PX_LAST'].iloc[split_point - 1]
    actual_prices = data[f'{commodity}_PX_LAST'][split_point:]
    predicted_prices = np.exp(predicted_log_returns.cumsum()) * last_train_price

    actual_direction = data['Direction'][split_point:]
    predicted_direction = (predicted_log_returns > 0).astype(int)

    combined_df = pd.concat([actual_prices, predicted_prices, test, predicted_log_returns, actual_direction, predicted_direction], axis=1)
    combined_df.columns = ['Price', 'Pred_Price', 'Log_Ret', 'Pred_Log_Ret', 'Dir', 'Pred_Dir']
    combined_df = combined_df.reset_index()
    combined_df.columns = ['Date', 'Price', 'Pred_Price', 'Log_Ret', 'Pred_Log_Ret', 'Dir', 'Pred_Dir']
    combined_df.set_index('Date', inplace=True)
    combined_df.index = pd.to_datetime(combined_df.index).date

    if combined_df.isna().any().any():
        print(f'Warning: Predictions for {commodity} contain NaN values.')

    return combined_df


arima = 0
garch = 1

for commodity in commodities:
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))

    if arima == 1:
        prediction_arima_df = auto_arima_mod(data, commodity=commodity, t_set_size=0.8, seasonal_period=1)
        prediction_arima_df.to_pickle(os.path.join(storage, 'predictions', 'arima', f'{commodity}_arima_pred.pkl'))

    if garch == 1:
        prediction_garch_df = auto_garch_mod(data, commodity=commodity, max_p=6, max_q=6, t_set_size=0.8)
        prediction_garch_df.to_pickle(os.path.join(storage, 'predictions', 'garch', f'{commodity}_garch_pred.pkl'))
