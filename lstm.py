import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
storage = os.path.join(directory, 'data')

commodities = ['CO1', 'HG1', 'S1', 'NG1', 'CT1', 'W1', 'GC1', 'LC1']
split_date = "2023-04-19"
gap_days = 0

for commodity in commodities:
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    data.dropna(inplace=True)
    data.index = pd.to_datetime(data.index)

    # predictors
    predictors = [f'{commodity}_Log_Price', f'{commodity}_MA_20',
                  f'{commodity}_SD_20', f'{commodity}_HL', f'{commodity}_OC',
                  f'{commodity}_OPEN_INT', f'{commodity}_PX_VOLUME', f'{commodity}_EMA_20',
                  f'{commodity}_MACD', f'{commodity}_RSI']

    # targets
    targets = [f'{commodity}_Log_Price_Tomorrow', f'{commodity}_Direction_Tomorrow']
    results = {}

    time_steps = 10
    batch_size = 32

    test_start_date = pd.to_datetime(split_date)
    gap_start_date = test_start_date - pd.DateOffset(days=gap_days)

    # train and test set
    train = data[data.index < gap_start_date]
    test = data[data.index >= test_start_date]
    gap = data[(data.index >= gap_start_date) & (data.index < test_start_date)]

    for target in targets:
        print(f"Processing target: {target}")

        y = data[target]
        if target == f'{commodity}_Log_Price_Tomorrow':
            y = y.values.reshape(-1, 1)
            model_type = 'regression'
            loss = 'mse'
            last_activation = 'linear'
            monitor_metric = 'val_loss'
        else:
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            y = y_encoded
            model_type = 'classification'
            loss = 'binary_crossentropy'
            last_activation = 'sigmoid'
            monitor_metric = 'val_accuracy'

        y_train = train[target]
        y_test = test[target]

        if model_type == 'regression':
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
            y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train[predictors])
        X_test_scaled = scaler.transform(test[predictors])

        train_generator = TimeseriesGenerator(X_train_scaled, y_train, length=time_steps, batch_size=batch_size)
        test_generator = TimeseriesGenerator(X_test_scaled, y_test, length=time_steps, batch_size=batch_size)

        model = Sequential([
            LSTM(175, input_shape=(time_steps, X_train_scaled.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(150),
            Dropout(0.2),
            Dense(100, activation='relu'),
            Dense(1 if model_type == 'regression' else 1, activation=last_activation)
        ])

        model.compile(optimizer='adam', loss=loss, metrics=['accuracy' if model_type == 'classification' else 'mse'])
        model.summary()

        callbacks = [EarlyStopping(monitor=monitor_metric, patience=70, restore_best_weights=True),
                     ModelCheckpoint(os.path.join(directory, 'best_model', f'best_model_{target}.keras'),
                                     monitor=monitor_metric, save_best_only=True)]

        model.fit(train_generator, epochs=200, validation_data=test_generator, callbacks=callbacks, verbose=1)

        predictions = model.predict(test_generator)
        if model_type == 'regression':
            predictions = scaler_y.inverse_transform(predictions)
        else:
            predictions = (predictions > 0.5).astype(int)

        adjusted_dates = test.index[time_steps:]
        actual_values = test.loc[adjusted_dates, target]

        if model_type == 'regression':
            results_df = pd.DataFrame({'Price': np.exp(actual_values),
                                       'Pred_Price': np.exp(predictions.flatten())}, index=adjusted_dates)
        else:
            actual_labels = [test.loc[date, target] for date in adjusted_dates]
            results_df = pd.DataFrame({'Dir': actual_labels,
                                       'Pred_Dir': predictions.flatten()}, index=adjusted_dates)

        results_df = results_df.shift(1)
        results_df.dropna(inplace=True)
        results[target] = results_df

    final_results = pd.concat(results.values(), axis=1)
    final_results.to_pickle(os.path.join(storage, 'predictions', 'lstm', f'{commodity}_lstm_pred.pkl'))