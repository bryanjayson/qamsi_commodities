import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
config = os.path.join(directory, 'config')
storage = os.path.join(directory, 'data')

commodities = ['CO1', 'HG1', 'S1', 'NG1', 'CT1', 'W1', 'GC1', 'LC1']
models = ['rf', 'lstm']


def plot_preds(commodity, model):
    path = os.path.join(storage, 'predictions', model, f'{commodity}_{model}_pred.pkl')
    if os.path.exists(path):
        df = pd.read_pickle(path)

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        rcParams['font.family'] = 'DejaVu Sans'

        plt.figure(figsize=(6, 4))
        plt.plot(df.index, df['Price'], 'b-', label='Actual', marker='', markersize=3, linewidth=1.5)
        plt.plot(df.index, df['Pred_Price'], 'r-', label='Predicted', marker='', markersize=3, linewidth=1.5)

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{commodity}-{model}: Actual vs. Predicted')
        plt.legend()

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.grid(True)

        plot_path = os.path.join(storage, 'plots', f'{commodity}_{model}_pred_plot.pdf')
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.close()


def accuracy(target, prediction):
    return accuracy_score(target, prediction)


def precision(target, prediction):
    return precision_score(target, prediction, zero_division=0)


def recall(target, prediction):
    return recall_score(target, prediction, zero_division=0)


def f1(target, prediction):
    return f1_score(target, prediction, zero_division=0)


def mse(target, prediction):
    return mean_squared_error(target, prediction)


def rmse(target, prediction):
    return np.sqrt(mean_squared_error(target, prediction))


def mape(target, prediction):
    return mean_absolute_percentage_error(target, prediction)


def mbe(target, prediction):
    return np.mean(prediction - target)


def r2(target, prediction):
    return r2_score(target, prediction)


results = {'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1', 'MSE', 'RMSE', 'MAPE', 'MBE', 'R2']}

for model in models:
    results[model] = {}

for commodity in commodities:
    print(commodity)
    for model in models:
        path = os.path.join(storage, 'predictions', model, f'{commodity}_{model}_pred.pkl')
        df = pd.read_pickle(path)

        if model == 'linear':
            accuracy_val = 0
            precision_val = 0
            recall_val = 0
            f1_val = 0
            mse_val = mse(df['Price'], df['Pred_Price'])
            rmse_val = rmse(df['Price'], df['Pred_Price'])
            mape_val = mape(df['Price'], df['Pred_Price'])
            mbe_val = mbe(df['Price'], df['Pred_Price'])
            r2_val = r2(df['Price'], df['Pred_Price'])
            results[model][commodity] = [accuracy_val, precision_val, recall_val, f1_val, mse_val, rmse_val,
                                                mape_val, mbe_val, r2_val]

            plot_preds(commodity, model)

        else:
            accuracy_val = accuracy(df['Dir'], df['Pred_Dir'])
            precision_val = precision(df['Dir'], df['Pred_Dir'])
            recall_val = recall(df['Dir'], df['Pred_Dir'])
            f1_val = f1(df['Dir'], df['Pred_Dir'])
            mse_val = mse(df['Price'], df['Pred_Price'])
            rmse_val = rmse(df['Price'], df['Pred_Price'])
            mape_val = mape(df['Price'], df['Pred_Price'])
            mbe_val = mbe(df['Price'], df['Pred_Price'])
            r2_val = r2(df['Price'], df['Pred_Price'])
            results[model][commodity] = [accuracy_val, precision_val, recall_val, f1_val, mse_val, rmse_val,
                                            mape_val, mbe_val, r2_val]

            plot_preds(commodity, model)

output = os.path.join(storage, 'performance', 'model_performance.xlsx')
with pd.ExcelWriter(output) as writer:
    for model in models:
        results_df = pd.DataFrame(results[model], index=results['Metrics'])
        results_df.to_excel(writer, sheet_name=model, index=True)
