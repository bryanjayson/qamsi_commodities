import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
storage = os.path.join(directory, 'data')


def buy_and_hold(commodity, start_date):
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    predictions = pd.read_pickle(os.path.join(storage, 'predictions', 'lstm', f'{commodity}_lstm_pred.pkl'))

    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)
    if not pd.api.types.is_datetime64_any_dtype(predictions.index):
        predictions.index = pd.to_datetime(predictions.index)

    if start_date:
        start_date = pd.to_datetime(start_date)
    else:
        start_date = predictions.index.min()

    filtered_data = data.loc[start_date:].copy()

    filtered_data['Cumulative_Log_Returns'] = filtered_data[f'{commodity}_Log_Returns'].cumsum()
    filtered_data['Cumulative_Returns'] = np.exp(filtered_data['Cumulative_Log_Returns']) - 1

    if not filtered_data.empty:
        filtered_data['Cumulative_Returns'].iloc[0] = 0

    result = filtered_data[['Cumulative_Returns']].copy()

    return result


def prediction_strat(commodity, model, start_date):
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    predictions = pd.read_pickle(os.path.join(storage, 'predictions', model, f'{commodity}_{model}_pred.pkl'))

    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)
    if not pd.api.types.is_datetime64_any_dtype(predictions.index):
        predictions.index = pd.to_datetime(predictions.index)

    start_date = pd.to_datetime(start_date)
    predictions = predictions.loc[start_date:]
    filtered_data = data.loc[start_date:].copy()

    predictions['Trade_Signal'] = predictions['Pred_Dir'].shift(-1)
    predictions['Log_Ret'] = filtered_data[f'{commodity}_Log_Returns']

    predictions['Daily_Log_Return'] = 0.0
    for i in range(1, len(predictions)):
        trade_signal = predictions.iloc[i - 1]['Trade_Signal']
        log_return = predictions.iloc[i]['Log_Ret']

        # trade signal
        if trade_signal == 1:
            predictions.iloc[i, predictions.columns.get_loc('Daily_Log_Return')] = log_return
        elif trade_signal == 0:
            predictions.iloc[i, predictions.columns.get_loc('Daily_Log_Return')] = 0

    # calculate cumulative returns from the start date
    predictions['Cumulative_Log_Return'] = predictions['Daily_Log_Return'].cumsum()
    predictions['Cumulative_Return'] = np.exp(predictions['Cumulative_Log_Return']) - 1

    return predictions[['Cumulative_Log_Return', 'Cumulative_Return']]


def plot_strat(commodity, start_date):
    buy_and_hold_data = buy_and_hold(commodity, start_date=start_date)
    pred_strat_lstm = prediction_strat(commodity, 'lstm', start_date=start_date)
    pred_strat_rf = prediction_strat(commodity, 'rf', start_date=start_date)

    rcParams['font.family'] = 'DejaVu Sans'

    plt.figure(figsize=(6, 4))

    plt.plot(buy_and_hold_data.index, buy_and_hold_data['Cumulative_Returns'] * 100,
             'b-', label='Buy and Hold', marker='', markersize=4, linewidth=1.5)

    plt.plot(pred_strat_lstm.index, pred_strat_lstm['Cumulative_Return'] * 100,
             'r-', label='LSTM', marker='', markersize=4, linewidth=1.5)
    plt.plot(pred_strat_rf.index, pred_strat_rf['Cumulative_Return'] * 100,
             'g-', label='RF', marker='', markersize=4, linewidth=1.5)

    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(storage, 'plots', f'{commodity}_strategies_comparison.pdf'))
    plt.close()


def plot_prices():
    rcParams['font.family'] = 'DejaVu Sans'

    for commodity in commodities:
        plt.figure(figsize=(6, 4))
        data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)

        plt.plot(data.index, data[f'{commodity}_PX_LAST'], 'b-', label=f'{commodity}', marker='', linewidth=1.5)

        plt.title(f'{commodity}: Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=24))

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(storage, 'plots', f'{commodity}_price_plot.pdf'))
        plt.close()


def plot_comparison(commodities, start_date):
    all_data = []

    for commodity in commodities:
        buy_hold = buy_and_hold(commodity, start_date)
        lstm_returns = prediction_strat(commodity, 'lstm', start_date)
        rf_returns = prediction_strat(commodity, 'rf', start_date)

        buy_hold_return = buy_hold['Cumulative_Returns'].iloc[-1] * 100
        lstm_return = lstm_returns['Cumulative_Return'].iloc[-1] * 100
        rf_return = rf_returns['Cumulative_Return'].iloc[-1] * 100

        all_data.append({
            'Commodity': commodity,
            'Buy and Hold': buy_hold_return,
            'LSTM': lstm_return,
            'RF': rf_return
        })

    results = pd.DataFrame(all_data)

    rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['blue', 'red', 'green']

    results.set_index('Commodity').plot(kind='bar', ax=ax, color=colors, width=0.8)

    ax.set_ylabel('Cumulative Return (%)')
    ax.set_xlabel('Commodity')
    ax.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    ax.legend(labels=['Buy and Hold', 'LSTM', 'RF'])

    plt.tight_layout()
    plt.savefig(os.path.join(storage, 'plots', 'strategies_comparison.pdf'))


def plot_prices_split(commodities, start_date=None):
    rcParams['font.family'] = 'DejaVu Sans'

    for commodity in commodities:
        plt.figure(figsize=(6, 4))
        data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)

        plt.plot(data.index, data[f'{commodity}_PX_LAST'], 'b-', label=f'{commodity}', marker='', linewidth=1.5)

        if start_date:
            start_date = pd.to_datetime(start_date)
            mask = data.index >= start_date
            plt.plot(data.index[mask], data[f'{commodity}_PX_LAST'][mask], 'r-', linewidth=1.5)

        plt.title(f'{commodity}: Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=24))

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(storage, 'plots', f'{commodity}_price_plot_split.pdf'))
        plt.close()


commodities = ['NG1']
start_date = "2023-05-01"


for commodity in commodities:

    buy_and_hold(commodity, start_date=start_date)
    prediction_strat(commodity, model='lstm', start_date=start_date)
    prediction_strat(commodity, model='rf', start_date=start_date)
    plot_strat(commodity, start_date=start_date)

plot_prices()
plot_comparison(['CO1', 'HG1', 'S1', 'NG1', 'CT1', 'W1', 'GC1', 'LC1'], start_date=start_date)
plot_prices_split(commodities, start_date=start_date)