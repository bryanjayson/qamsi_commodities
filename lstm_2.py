import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)
storage = os.path.join(directory, 'data')

commodities = ['LC1']

for commodity in commodities:
    data = pd.read_pickle(os.path.join(storage, 'raw', f'{commodity}.pkl'))
    data.dropna(inplace=True)

    predictors = [f'{commodity}_MA_5', f'{commodity}_MA_10', f'{commodity}_MA_15', f'{commodity}_SD_5',
                  f'{commodity}_SD_10', f'{commodity}_SD_15', f'{commodity}_HL', f'{commodity}_OC',
                  f'{commodity}_OPEN_INT', f'{commodity}_PX_VOLUME', f'{commodity}_PX_LAST', f'{commodity}_PX_HIGH',
                  f'{commodity}_PX_LOW', f'{commodity}_PX_OPEN', f'{commodity}_EMA_10', f'{commodity}_EMA_20',
                  f'{commodity}_MACD', f'{commodity}_RSI']

    X = data[predictors]
    dates = data.index

    # Define targets
    targets = ['Price_Tomorrow', 'Direction_Tomorrow']
    results = {}

    time_steps = 30
    batch_size = 8

    for target in targets:
        print(f"Processing target: {target}")

        y = data[target]
        if target == 'Price_Tomorrow':
            y = y.values.reshape(-1, 1)
            model_type = 'regression'
            loss = 'mse'
            last_activation = 'linear'
            monitor_metric = 'val_loss'
        else:
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            y = to_categorical(y_encoded)
            model_type = 'classification'
            loss = 'categorical_crossentropy'
            last_activation = 'softmax'
            monitor_metric = 'val_accuracy'

        # Split data and maintain date indices
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, shuffle=False)

        # Feature scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == 'regression':
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            y_train = scaler_y.fit_transform(y_train)
            y_test = scaler_y.transform(y_test)

        # Create generators
        train_generator = TimeseriesGenerator(X_train_scaled, y_train, length=time_steps, batch_size=batch_size)
        test_generator = TimeseriesGenerator(X_test_scaled, y_test, length=time_steps, batch_size=batch_size)

        # Model architecture
        model = Sequential([
            LSTM(50, input_shape=(time_steps, X_train_scaled.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(y.shape[1], activation=last_activation)])

        model.compile(optimizer='adam', loss=loss, metrics=['accuracy' if model_type == 'classification' else 'mse'])
        model.summary()

        callbacks = [
            EarlyStopping(monitor=monitor_metric, patience=10, restore_best_weights=True),
            ModelCheckpoint(f'best_model_{target}.keras', monitor=monitor_metric, save_best_only=True)]

        model.fit(train_generator, epochs=3, validation_data=test_generator, callbacks=callbacks, verbose=1)

        predictions = model.predict(test_generator)
        if model_type == 'regression':
            predictions = scaler_y.inverse_transform(predictions)

        if model_type == 'classification':
            predictions = np.argmax(predictions, axis=1)

        # Correct date indexing for results
        adjusted_dates = dates_test[time_steps:]

        # Retrieve actual values directly from the original DataFrame
        actual_values = data.loc[adjusted_dates, target]

        # Result DataFrame creation
        if model_type == 'regression':
            results_df = pd.DataFrame({'Actual Price': actual_values, 'Predicted Price': predictions.flatten()},
                                      index=adjusted_dates)
        else:
            actual_labels = [data.loc[date, target] for date in
                             adjusted_dates]  # Retrieving original categorical labels
            results_df = pd.DataFrame({'Actual Direction': actual_labels, 'Predicted Direction': predictions},
                                      index=adjusted_dates)

        results[target] = results_df

    final_results = pd.concat(results.values(), axis=1)
    final_results.to_pickle(os.path.join(storage, 'predictions', 'lstm', f'{commodity}_lstm_pred.pkl'))