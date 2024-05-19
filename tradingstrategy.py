import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
storage = os.path.join(directory, 'data')

commodities = ['CO1', 'HG1', 'S1', 'NG1', 'W1']

def buy_and_hold(commodity):
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    predictions = pd.read_pickle(os.path.join(storage, 'predictions', 'lstm', f'{commodity}_lstm_pred.pkl'))

    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)
    if not pd.api.types.is_datetime64_any_dtype(predictions.index):
        predictions.index = pd.to_datetime(predictions.index)

    start_date = predictions.index.min()
    filtered_data = data.loc[start_date:].copy()

    filtered_data['Cumulative_Log_Returns'] = filtered_data[f'{commodity}_Log_Returns'].cumsum()
    filtered_data['Cumulative_Returns'] = np.exp(filtered_data['Cumulative_Log_Returns']) - 1

    if not filtered_data.empty:
        filtered_data['Cumulative_Returns'].iloc[0] = 0

    result = filtered_data[['Cumulative_Returns']].copy()

    return result


def prediction_strat(commodity, model):
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    predictions = pd.read_pickle(os.path.join(storage, 'predictions', model, f'{commodity}_{model}_pred.pkl'))

    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)
    if not pd.api.types.is_datetime64_any_dtype(predictions.index):
        predictions.index = pd.to_datetime(predictions.index)

    start_date = predictions.index.min()
    filtered_data = data.loc[start_date:].copy()

    predictions['Trade_Signal'] = predictions['Pred_Dir'].shift(-1)
    predictions['Log_Ret'] = filtered_data[f'{commodity}_Log_Returns']

    predictions['Daily_Log_Return'] = 0.0
    for i in range(1, len(predictions)):
        trade_signal = predictions.iloc[i - 1]['Trade_Signal']
        log_return = predictions.iloc[i]['Log_Ret']

        if trade_signal == 1:
            predictions.iloc[i, predictions.columns.get_loc('Daily_Log_Return')] = log_return
        elif trade_signal == 0:
            predictions.iloc[i, predictions.columns.get_loc('Daily_Log_Return')] = -log_return

    predictions['Cumulative_Log_Return'] = predictions['Daily_Log_Return'].cumsum()
    predictions['Cumulative_Return'] = np.exp(predictions['Cumulative_Log_Return']) - 1

    return predictions[['Cumulative_Log_Return', 'Cumulative_Return']]

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import pandas as pd
import os

def plot_strat(commodity):
    buy_and_hold_data = buy_and_hold(commodity)
    pred_strat_lstm = prediction_strat(commodity, 'lstm')
    pred_strat_rf = prediction_strat(commodity, 'rf')

    # Set font for the plot
    rcParams['font.family'] = 'DejaVu Sans'

    # Create a plot
    plt.figure(figsize=(12, 8))

    # Plotting the cumulative returns from buy and hold strategy
    plt.plot(buy_and_hold_data.index, buy_and_hold_data['Cumulative_Returns'] * 100,
             'b-', label='Buy and Hold', marker='o', markersize=4, linewidth=1)

    # Plotting the cumulative returns from the prediction strategies
    plt.plot(pred_strat_lstm.index, pred_strat_lstm['Cumulative_Return'] * 100,
             'r-', label='LSTM', marker='x', markersize=4, linewidth=1)
    plt.plot(pred_strat_rf.index, pred_strat_rf['Cumulative_Return'] * 100,
             'g-', label='RF', marker='s', markersize=4, linewidth=1)


    # Setting the title, labels, and legends
    plt.title(f'{commodity} Trading Strategies Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()

    # Formatting the date on the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Adding grid for better readability
    plt.grid(True)

    # Layout adjustment to prevent clipping of labels
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(os.path.join(storage, 'plots', f'{commodity}_strategies_comparison.pdf'))

    # Close the plot environment to free up memory
    plt.close()



for commodity in commodities:
    buy_and_hold(commodity)
    prediction_strat(commodity, model='lstm')
    prediction_strat(commodity, model='rf')
    plot_strat(commodity)


print()