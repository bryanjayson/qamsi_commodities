import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
config = os.path.join(directory, 'config')
storage = os.path.join(directory, 'data')

commodities = ['CO1', 'HG1', 'S1', 'NG1', 'CT1', 'W1', 'GC1', 'LC1']

n_est = 8000
gap_days = 0

model_class = RandomForestClassifier(n_estimators=n_est, min_samples_split=4, max_depth=4,
                                     random_state=4, criterion="entropy")
model_reg = RandomForestRegressor(n_estimators=n_est, min_samples_split=4, max_depth=10,
                                  random_state=4, criterion="squared_error")


def predict_dir(commodity, train, test, predictors, model, confidence=0.5):
    model.fit(train[predictors], train[f'{commodity}_Direction_Tomorrow'])
    predictions_dir = model.predict_proba(test[predictors])[:, 1]
    predictions_dir[predictions_dir >= confidence] = 1
    predictions_dir[predictions_dir < confidence] = 0
    predictions_dir = pd.Series(predictions_dir, index=test.index, name=f'{commodity}_Predictions_Dir')
    combined_dir = pd.concat([test[f'{commodity}_Direction'], predictions_dir], axis=1)
    return combined_dir


# not used in final paper
def predict_price(commodity, train, test, predictors, model):
    model.fit(train[predictors], train[f'{commodity}_Price_Tomorrow'])
    predictions_price = pd.Series(model.predict(test[predictors]), index=test.index, name="Predictions_Price")
    combined_price = pd.concat([test[f'{commodity}_PX_LAST'], predictions_price], axis=1)
    return combined_price



def predict_ret(commodity, train, test, predictors, model):
    model.fit(train[predictors], train[f'{commodity}_Ret_Tomorrow'])
    predictions_return = model.predict(test[predictors])
    predictions_return = pd.Series(predictions_return, index=test.index, name='Predictions_Ret')
    combined_ret = pd.concat([test[f'{commodity}_Log_Returns'], predictions_return], axis=1)
    return combined_ret


def rf_fit(commodity, data, model_class, model_reg, predictors, split_date, confidence=0.50):
    data.index = pd.to_datetime(data.index)

    split_date = pd.to_datetime(split_date)
    gap_start_date = split_date - pd.DateOffset(days=gap_days)

    # split the dataset into training, gap, and testing sets
    train = data[data.index < gap_start_date]
    print(len(train))
    gap = data[(data.index >= gap_start_date) & (data.index < split_date)]
    test = data[data.index >= split_date]

    print("Predicting Direction")
    predictions_dir = predict_dir(commodity, train, test, predictors, model_class, confidence=confidence)
    print("Predicting Price")
    predictions_price = predict_price(commodity, train, test, predictors, model_reg)
    predictions_all = pd.concat([predictions_price, predictions_dir], axis=1)

    rename = ['Price', 'Pred_Price', 'Dir', 'Pred_Dir']
    predictions_all.columns = rename

    storage_path = os.path.join(storage, 'predictions', 'rf', f'{commodity}_rf_pred.pkl')
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    predictions_all.to_pickle(storage_path)

    return predictions_all, gap


for commodity in commodities:
    print(commodity)
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    data.dropna()

    predictors = [f'{commodity}_PX_LAST', f'{commodity}_MA_20',
                  f'{commodity}_SD_20', f'{commodity}_HL', f'{commodity}_OC',
                  f'{commodity}_OPEN_INT', f'{commodity}_PX_VOLUME', f'{commodity}_EMA_20',
                  f'{commodity}_MACD', f'{commodity}_RSI']

    rf_predictions = rf_fit(commodity, data, model_class, model_reg,
                            predictors, split_date="2023-05-01", confidence=0.5)