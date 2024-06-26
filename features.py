import numpy as np
import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
config = os.path.join(directory, 'config')
storage = os.path.join(directory, 'data')

data_all = pd.read_excel(os.path.join(storage, 'raw', 'commodity_data.xlsx'))
data_all = data_all.iloc[::-1]

data_all['Dates'] = pd.to_datetime(data_all['Dates'])
data_all['Dates'] = data_all['Dates'].dt.strftime('%Y-%m-%d')
data_all.set_index('Dates', inplace=True)

commodities = ['CO1', 'HG1', 'S1', 'NG1', 'CT1', 'W1', 'GC1', 'LC1']
column_ranges = {
    'LC1': list(range(0, 6)),
    'CO1': list(range(6, 12)),
    'CL1': list(range(12, 18)),
    'CT1': list(range(18, 24)),
    'NG1': list(range(24, 30)),
    'HG1': list(range(30, 36)),
    'W1': list(range(36, 42)),
    'GC1': list(range(42, 48)),
    'S1': list(range(48, 54))}


def test_stationarity(dataframe):
    results = {}
    for column in dataframe.columns:
        test_result = adfuller(dataframe[column].dropna())

        p_value = test_result[1]
        is_stationary = p_value < 0.05
        results[column] = is_stationary

        print(f"Column: {column}, Stationary: {is_stationary}")

    return results


for commodity in commodities:
    print(commodity)
    cols = data_all.columns[column_ranges[commodity]].tolist()
    data = data_all.loc[:, cols]

    # Setting target (1 means price tomorrow is higher than price today), calculating returns
    data[f'{commodity}_Log_Price'] = np.log(data[f'{commodity}_PX_LAST'])
    data[f'{commodity}_Price_Tomorrow'] = data[f'{commodity}_PX_LAST'].shift(-1)
    data[f'{commodity}_Log_Price_Tomorrow'] = np.log(data[f'{commodity}_Log_Price'])
    data[f'{commodity}_Direction_Tomorrow'] = (data[f'{commodity}_Price_Tomorrow'] >
                                               data[f'{commodity}_PX_LAST']).astype(int)
    data[f'{commodity}_Direction'] = data[f'{commodity}_Direction_Tomorrow'].shift(1)
    data[f'{commodity}_Log_Returns'] = np.log(data[f'{commodity}_PX_LAST'] / data[f'{commodity}_PX_LAST'].shift(1))
    data[f'{commodity}_Ret_Tomorrow'] = data[f'{commodity}_Log_Returns'].shift(-1)

    horizons = [10, 20]

    for horizon in horizons:
        moving_average = data.rolling(horizon, closed='right').mean()
        col_name = f'{commodity}_MA_{horizon}'

        data[col_name] = moving_average[f'{commodity}_PX_LAST']
        data[f'{commodity}_SD_{horizon}'] = data[f'{commodity}_PX_LAST'].rolling(horizon, closed='right').std()

    # calculate RSI
    delta = -data[f'{commodity}_PX_LAST'].diff(-1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14, closed='right').mean()
    avg_loss = loss.rolling(window=14, min_periods=14, closed='right').mean()

    rs = avg_gain / avg_loss
    data[f'{commodity}_RSI'] = 100 - (100 / (1 + rs))

    data[f'{commodity}_Return_Lag_1'] = data[f'{commodity}_Log_Returns'].shift(1)
    data[f'{commodity}_Return_Lag_2'] = data[f'{commodity}_Log_Returns'].shift(2)
    data[f'{commodity}_EMA_10'] = data[f'{commodity}_PX_LAST'].ewm(span=10, adjust=False).mean()
    data[f'{commodity}_EMA_20'] = data[f'{commodity}_PX_LAST'].ewm(span=20, adjust=False).mean()
    data[f'{commodity}_MACD'] = data[f'{commodity}_EMA_10'] - data[f'{commodity}_EMA_20']
    data[f'{commodity}_HL'] = data[f'{commodity}_PX_HIGH'] - data[f'{commodity}_PX_LOW']
    data[f'{commodity}_OC'] = data[f'{commodity}_PX_OPEN'] - data[f'{commodity}_PX_LAST']

    data.to_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))

    test_stationarity(data)

summary_data = []

for commodity in commodities:
    file_path = os.path.join(storage, 'raw', f'{commodity}.pkl')
    data = pd.read_pickle(file_path)

    summary_stats = {
        'Commodity': commodity,
        'Count': data[f'{commodity}_PX_LAST'].count(),
        'Mean': data[f'{commodity}_PX_LAST'].mean(),
        'Minimum': data[f'{commodity}_PX_LAST'].min(),
        'Maximum': data[f'{commodity}_PX_LAST'].max(),
        'Standard Deviation': data[f'{commodity}_PX_LAST'].std(),
        'Mean OI': data[f'{commodity}_OPEN_INT'].mean(),
        'Mean Volume': data[f'{commodity}_PX_VOLUME'].mean()
    }
    summary_data.append(summary_stats)

summary_df = pd.DataFrame(summary_data)

summary_df.to_excel(os.path.join(storage, 'summary_statistics.xlsx'))