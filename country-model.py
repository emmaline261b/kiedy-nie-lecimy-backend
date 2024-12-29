import pickle
import pandas as pd
import numpy as np
import os

import tensorflow
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

#optimizery
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers import Nadam




from sklearn.preprocessing import MinMaxScaler




country_df = pd.read_csv('time_series_by_country/time_series_Afghanistan.csv')
print('wczytano plik')
country_df['Date'] = pd.to_datetime(country_df['Date'])


# Podział na dane treningowe i testowe
training_data = country_df[country_df['Date'].dt.year.between(2000, 2022)]
testing_data = country_df[country_df['Date'].dt.year.between(2023, 2024)]

# Przygotowanie X i y dla danych treningowych
X_train = training_data.drop(columns=['Date', 'Is_disaster']).values
y_train = training_data['Is_disaster'].values

# Przygotowanie X i y dla danych testowych
X_test = testing_data.drop(columns=['Date', 'Is_disaster']).values
y_test = testing_data['Is_disaster'].values

# Tworzenie sekwencji
def create_sequences(features, target, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

# Parametr: długość sekwencji
sequence_length = 365

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('przeskalowano dane cech.')

# Tworzenie sekwencji na danych znormalizowanych
X_train_lstm, y_train_lstm = create_sequences(X_train_scaled, y_train, sequence_length)
X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test, sequence_length)

print('utworzono sekwencje na danych treningowych i testowych')

# Konwersja danych na float32
X_train_lstm = X_train_lstm.astype('float32')
y_train_lstm = y_train_lstm.astype('float32')
X_test_lstm = X_test_lstm.astype('float32')
y_test_lstm = y_test_lstm.astype('float32')

# Sprawdzenie danych treningowych
assert not np.isnan(X_train_lstm).any(), "X_train_lstm contains NaN values"
assert not np.isnan(y_train_lstm).any(), "y_train_lstm contains NaN values"
assert not np.isnan(X_test_lstm).any(), "X_test_lstm contains NaN values"
assert not np.isnan(y_test_lstm).any(), "y_test_lstm contains NaN values"
assert not np.isinf(X_train_lstm).any(), "X_train_lstm contains infinite values"
assert not np.isinf(y_train_lstm).any(), "y_train_lstm contains infinite values"

print("Min X_train_lstm:", np.min(X_train_lstm))
print("Max X_train_lstm:", np.max(X_train_lstm))
print("Min y_train_lstm:", np.min(y_train_lstm))
print("Max y_train_lstm:", np.max(y_train_lstm))

# Ważenie klas
class_weights = {
    0: len(y_train_lstm) / np.sum(y_train_lstm == 0),
    1: len(y_train_lstm) / np.sum(y_train_lstm == 1),
}

print("Wagi klas:", class_weights)
assert all(weight > 0 for weight in class_weights.values()), "class_weights contain non-positive values"


# Budowa modelu LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(100, activation='relu', return_sequences=True,
                    input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model_lstm.add(Dropout(0.2))
# model_lstm.add(BatchNormalization())
model_lstm.add(LSTM(50, activation='relu', return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(20, activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1, activation='sigmoid'))

# Kompilacja modelu
optimizer = Adam(learning_rate=0.000001, clipvalue=1.0)
optimizer2 = RMSprop(learning_rate=0.0001)
optimizer3 = Adagrad(learning_rate=0.0001)
optimizer4 = AdamW(learning_rate=0.0001, weight_decay=1e-5)
optimizer5 = Nadam(learning_rate=0.0001)


model_lstm.compile(optimizer=optimizer2, loss='binary_crossentropy', metrics=['accuracy'])
print('zbudowano i skompilowano model')

# Trenowanie modelu
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_lstm.fit(
    X_train_lstm, y_train_lstm,
    epochs=100, batch_size=128,
    validation_data=(X_test_lstm, y_test_lstm),
    callbacks=[early_stopping]
)
print('przetrenowano model')

# Prognoza i ocena
predictions_test = model_lstm.predict(X_test_lstm)
assert not np.isnan(predictions_test).any(), "Predictions contain NaN values"


predictions_binary_test = (predictions_test > 0.5).astype(int)
print(classification_report(y_test_lstm, predictions_binary_test))
print("AUC-ROC:", roc_auc_score(y_test_lstm, predictions_test))



def save_model_to_pkl(model, country_name, folder_name):
    """
    Zapisuje model LSTM do pliku .pkl w określonym folderze i wyświetla komunikat po zapisaniu.

    :param model: Model LSTM do zapisania
    :param country_name: Nazwa kraju, używana w nazwie pliku
    :param folder_name: Nazwa folderu, w którym model zostanie zapisany (domyślnie 'model_by_country')
    """
    # Tworzenie folderu, jeśli nie istnieje
    os.makedirs(folder_name, exist_ok=True)

    # Ścieżka do pliku
    file_name = f"{country_name.lower().replace(' ', '_')}_model.pkl"
    file_path = os.path.join(folder_name, file_name)

    try:
        # Zapisanie modelu do pliku .pkl
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

        # Wyświetlenie komunikatu sukcesu
        print(f"Model został zapisany w pliku '{file_path}'")
    except Exception as e:
        # Wyświetlenie błędu, jeśli coś pójdzie nie tak
        print(f"Wystąpił błąd podczas zapisywania modelu: {e}")



save_model_to_pkl(model_lstm, country_name='Afghanistan', folder_name='model_by_country')
