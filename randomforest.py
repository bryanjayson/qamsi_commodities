import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
config = os.path.join(directory, 'config')
storage = os.path.join(directory, 'data')

commodities = ['LC1', 'CO1', 'CT1', 'NG1', 'HG1', 'W1', 'GC1', 'S1']

n_est = 500

model_class = RandomForestClassifier(n_estimators=n_est, min_samples_split=2, max_depth=20, random_state=4)
model_reg = RandomForestRegressor(n_estimators=n_est, min_samples_split=2, max_depth=20, random_state=4)


def predict_dir(train, test, predictors, model, confidence=0.6):
    model.fit(train[predictors], train['Direction_Tomorrow'])
    predictions_dir = model.predict_proba(test[predictors])[:, 1]
    predictions_dir[predictions_dir >= confidence] = 1
    predictions_dir[predictions_dir < confidence] = 0
    predictions_dir = pd.Series(predictions_dir, index=test.index, name='Predictions_Dir')
    combined_dir = pd.concat([test['Direction'], predictions_dir], axis=1)
    return combined_dir


def predict_price(train, test, predictors, model):
    model.fit(train[predictors], train['Price_Tomorrow'])
    predictions_price = model.predict(test[predictors])
    predictions_price = pd.Series(predictions_price, index=test.index, name='Predictions_Price')
    combined_price = pd.concat([test['Price_Tomorrow'], predictions_price], axis=1)
    return combined_price


def predict_ret(train, test, predictors, model):
    model.fit(train[predictors], train['Ret_Tomorrow'])
    predictions_return = model.predict(test[predictors])
    predictions_return = pd.Series(predictions_return, index=test.index, name='Predictions_Ret')
    combined_ret = pd.concat([test['Ret_Tomorrow'], predictions_return], axis=1)
    return combined_ret


def rf_fit(data, model_class, model_reg, predictors, t_set_size=0.8, confidence=0.50):
    split_point = int(len(data) * t_set_size)
    train, test = data[:split_point], data[split_point:]

    predictions_dir = predict_dir(train, test, predictors, model_class, confidence=confidence)
    predictions_price = predict_price(train, test, predictors, model_reg)
    predictions_ret = predict_ret(train, test, predictors, model_reg)
    predictions_all = pd.concat([predictions_price, predictions_ret, predictions_dir], axis=1)

    rename = ['Price', 'Pred_Price', 'Log_Ret', 'Pred_Log_Ret', 'Dir', 'Pred_Dir']
    predictions_all.columns = rename

    predictions_all.to_pickle(os.path.join(storage, 'predictions', 'rf', f'{commodity}_rf_pred.pkl'))

    return predictions_all


for commodity in commodities:
    print(commodity)
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    data.dropna()

    predictors = [f'{commodity}_MA_5', f'{commodity}_MA_10', f'{commodity}_MA_15', f'{commodity}_SD_5',
                  f'{commodity}_SD_10', f'{commodity}_SD_15', f'{commodity}_HL', f'{commodity}_OC',
                  f'{commodity}_OPEN_INT', f'{commodity}_PX_VOLUME', f'{commodity}_PX_LAST', f'{commodity}_PX_HIGH',
                  f'{commodity}_PX_LOW', f'{commodity}_PX_OPEN', f'{commodity}_PX_Lag_1', f'{commodity}_PX_Lag_2',
                  f'{commodity}_EMA_10', f'{commodity}_EMA_20', f'{commodity}_MACD', f'{commodity}_RSI']

    rf_predictions = rf_fit(data, model_class, model_reg, predictors, t_set_size=0.94, confidence=0.6)
