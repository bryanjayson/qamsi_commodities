import numpy as np
import pandas as pd
import os

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
config = os.path.join(directory, 'config')
storage = os.path.join(directory, 'data')

data_all = pd.read_excel(os.path.join(storage, 'raw', 'commodity_data.xlsx'))
data_all = data_all.iloc[::-1]

data_all['Dates'] = pd.to_datetime(data_all['Dates'])
data_all['Dates'] = data_all['Dates'].dt.strftime('%Y-%m-%d')
data_all.set_index('Dates', inplace=True)

commodities = ['LC1', 'CO1', 'CT1', 'NG1', 'HG1', 'W1', 'GC1', 'S1']
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

for commodity in commodities:
    cols = data_all.columns[column_ranges[commodity]].tolist()
    data = data_all.loc[:, cols]

    # Setting target (1 means price tomorrow is higher than price today), calculating returns
    data['Price_Tomorrow'] = data[f'{commodity}_PX_LAST'].shift(-1)
    data['Direction_Tomorrow'] = (data['Price_Tomorrow'] > data[f'{commodity}_PX_LAST']).astype(int)
    data['Direction'] = data['Direction_Tomorrow'].shift(1)
    data['Log_Returns'] = np.log(data[f'{commodity}_PX_LAST'] / data[f'{commodity}_PX_LAST'].shift(1))
    data['Ret_Tomorrow'] = data['Log_Returns'].shift(-1)

    horizons = [7, 14, 21]

    for horizon in horizons:
        moving_average = data.rolling(horizon).mean()
        col_name = f'{commodity}_MA_{horizon}'

        data[col_name] = moving_average[f'{commodity}_PX_LAST']
        data[f'{commodity}_SD_{horizon}'] = data[f'{commodity}_PX_LAST'].rolling(horizon).std()

    data[f'{commodity}_PX_Lag_1'] = data[f'{commodity}_PX_LAST'].shift(1)
    data[f'{commodity}_PX_Lag_2'] = data[f'{commodity}_PX_LAST'].shift(2)
    data[f'{commodity}_EMA'] = data[f'{commodity}_PX_LAST'].ewm(span=5).mean()
    data[f'{commodity}_HL'] = data[f'{commodity}_PX_HIGH'] - data[f'{commodity}_PX_LOW']
    data[f'{commodity}_OC'] = data[f'{commodity}_PX_OPEN'] - data[f'{commodity}_PX_LAST']

    data.dropna(inplace=True)

    data.to_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    print(commodity)


#### IMPLEMENT SCALING FOR NN

print()