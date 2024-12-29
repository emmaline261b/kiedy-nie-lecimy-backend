import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# lstm
import numpy as np
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

wczytany_plik = pd.read_csv('resources/zastosowanie-ai-katastrofy.csv')
print('wczytano plik')

columns_to_delete = ['Classification Key', 'Disaster Group', 'ISO', 'Location', 'Associated Types', 'Origin',
                     'No. Homeless', 'DisNo.', 'Historic', 'External IDs', 'Event Name', 'Entry Date', 'Last Update',
                     'Total Deaths', 'No. Injured', 'No. Affected', 'Total Affected']
dane = wczytany_plik.drop(columns=columns_to_delete)

# Filtrujemy dane
filtered_data = dane[
    dane['Start Month'].isnull() &
    dane['End Month'].notna()]

def fill_start_month_for_row(row):
    if pd.isna(row['Start Month']) and pd.notna(row['End Month']):
        # Filtrujemy dane dla tego samego typu katastrofy i subregionu
        relevant_data = dane[
            (dane['Disaster Type'] == row['Disaster Type']) &
            (dane['Subregion'] == row['Subregion']) &
            (dane['End Month'] <= row['End Month']) &  # Warunek End Month
            (dane['Start Month'].notna())  # Tylko wiersze z istniejącym Start Month
            ]

        if len(relevant_data) < 3:
            relevant_data = dane[
                (dane['Disaster Type'] == row['Disaster Type']) &
                (dane['Region'] == row['Region']) &
                (dane['End Month'] <= row['End Month']) &  # Warunek End Month
                (dane['Start Month'].notna())  # Tylko wiersze z istniejącym Start Month
                ]

        # Obliczamy medianę Start Month z przefiltrowanych danych
        median_start_month = relevant_data['Start Month'].median()
        if pd.notna(median_start_month):  # Sprawdzamy, czy mediana istnieje
            row['Start Month'] = int(median_start_month)
        return row

# Zastosowanie funkcji do wszystkich wierszy
updated_data = filtered_data.apply(fill_start_month_for_row, axis=1)

dane.update(updated_data)

# Filtrujemy dane z brakującym 'End Month' i istniejącym 'Start Month'
filtered_data_end_month = dane[
    dane['End Month'].isnull() &
    dane['Start Month'].notna()
    ]


def fill_end_month_for_row(row):
    if pd.isna(row['End Month']) and pd.notna(row['Start Month']):

        # Filtrujemy dane dla tego samego typu katastrofy i subregionu
        relevant_data = dane[
            (dane['Disaster Type'] == row['Disaster Type']) &
            (dane['Subregion'] == row['Subregion']) &
            (dane['Start Month'] >= row['Start Month']) &  # Warunek Start Month
            (dane['End Month'].notna())  # Tylko wiersze z istniejącym End Month
            ]

        if len(relevant_data) < 4:
            relevant_data = dane[
                (dane['Disaster Type'] == row['Disaster Type']) &
                (dane['Region'] == row['Region']) &
                (dane['Start Month'] >= row['Start Month']) &  # Warunek Start Month
                (dane['End Month'].notna())  # Tylko wiersze z istniejącym End Month
                ]

        if len(relevant_data) < 4:
            relevant_data = dane[
                (dane['Disaster Type'] == row['Disaster Type']) &
                (dane['Start Month'] >= row['Start Month']) &  # Warunek Start Month
                (dane['End Month'].notna())  # Tylko wiersze z istniejącym End Month
                ]

        # Obliczamy medianę End Month z przefiltrowanych danych
        median_end_month = relevant_data['End Month'].median()

        if pd.notna(median_end_month):  # Sprawdzamy, czy mediana istnieje
            row['End Month'] = int(median_end_month)
    return row


# Zastosowanie funkcji tylko do przefiltrowanych wierszy
updated_data_end_month = filtered_data_end_month.apply(fill_end_month_for_row, axis=1)

# Aktualizacja oryginalnych danych
dane.update(updated_data_end_month)

# Filtrujemy dane z brakującym 'Start Day' i znanym 'End Month'
filtered_data_missing_day = dane[
    dane['Start Day'].isna() &
    dane['End Month'].notna()
    ]


def fill_start_day_for_row(row):
    relevant_data = None

    if pd.isna(row['Start Day']) and pd.notna(row['End Month']):

        if pd.isna(row['End Day']):  # with no end day
            relevant_data = dane[
                (dane['Disaster Type'] == row['Disaster Type']) &
                (dane['Subregion'] == row['Subregion']) &
                (dane['End Month'] <= row['End Month']) &
                (dane['Start Day'].notna())
                ]

            if len(relevant_data) < 4:
                relevant_data = dane[
                    (dane['Disaster Type'] == row['Disaster Type']) &
                    (dane['Region'] == row['Region']) &
                    (dane['End Month'] <= row['End Month']) &
                    (dane['Start Day'].notna())
                    ]

            if len(relevant_data) < 4:
                relevant_data = dane[
                    (dane['Disaster Type'] == row['Disaster Type']) &
                    (dane['End Month'] <= row['End Month']) &
                    (dane['Start Day'].notna())
                    ]


        elif pd.notna(row['End Day']):  # with end day
            relevant_data = dane[
                (dane['Disaster Type'] == row['Disaster Type']) &
                (dane['Subregion'] == row['Subregion']) &
                ((dane['End Month'] < row['End Month']) |
                 ((dane['End Month'] == row['End Month']) & (dane['End Day'] <= row['End Day']))) &
                (dane['Start Day'].notna())
                ]

            if len(relevant_data) < 4:
                relevant_data = dane[
                    (dane['Disaster Type'] == row['Disaster Type']) &
                    (dane['Region'] == row['Region']) &
                    ((dane['End Month'] < row['End Month']) |
                     ((dane['End Month'] == row['End Month']) & (dane['End Day'] < row['End Day']))) &
                    (dane['Start Day'].notna())
                    ]

            if len(relevant_data) < 4:
                relevant_data = dane[
                    (dane['Disaster Type'] == row['Disaster Type']) &
                    ((dane['End Month'] < row['End Month']) |
                     ((dane['End Month'] == row['End Month']) & (dane['End Day'] < row['End Day']))) &
                    (dane['Start Day'].notna())
                    ]

        # Obliczamy medianę 'Start Day' z przefiltrowanych danych
        median_start_day = relevant_data['Start Day'].median()

        # jeśli za mało danych, bierzemy ostatni dzień miesiąca
        if len(relevant_data) < 4:
            median_start_day = 1

        # Uzupełniamy 'Start Day', jeśli wyznaczono medianę
        if pd.notna(median_start_day):
            row['Start Day'] = int(median_start_day)

    return row


# Zastosowanie funkcji tylko do przefiltrowanych wierszy
updated_data = filtered_data_missing_day.apply(fill_start_day_for_row, axis=1)

# Aktualizacja oryginalnych danych
dane.update(updated_data)

# Filtrujemy dane z brakującym 'End Day' i znanym 'Start Month'
filtered_data_missing_end_day = dane[
    dane['End Day'].isna() &
    dane['Start Month'].notna()
    ]


def fill_end_day_for_row(row):
    relevant_data = None

    if pd.isna(row['End Day']) and pd.notna(row['Start Month']):

        if pd.isna(row['Start Day']):  # with no start day
            relevant_data = dane[
                (dane['Disaster Type'] == row['Disaster Type']) &
                (dane['Subregion'] == row['Subregion']) &
                (dane['Start Month'] >= row['Start Month']) &
                (dane['End Day'].notna())
                ]

            if len(relevant_data) < 4:
                relevant_data = dane[
                    (dane['Disaster Type'] == row['Disaster Type']) &
                    (dane['Region'] == row['Region']) &
                    (dane['Start Month'] >= row['Start Month']) &
                    (dane['End Day'].notna())
                    ]

            if len(relevant_data) < 4:
                relevant_data = dane[
                    (dane['Disaster Type'] == row['Disaster Type']) &
                    (dane['Start Month'] >= row['Start Month']) &
                    (dane['End Day'].notna())
                    ]


        elif pd.notna(row['Start Day']):  # with start day
            relevant_data = dane[
                (dane['Disaster Type'] == row['Disaster Type']) &
                (dane['Subregion'] == row['Subregion']) &
                ((dane['Start Month'] > row['Start Month']) |
                 ((dane['Start Month'] == row['Start Month']) & (dane['Start Day'] >= row['Start Day']))) &
                (dane['End Day'].notna())
                ]

            if len(relevant_data) < 4:
                relevant_data = dane[
                    (dane['Disaster Type'] == row['Disaster Type']) &
                    (dane['Region'] == row['Region']) &
                    ((dane['Start Month'] > row['Start Month']) |
                     ((dane['Start Month'] == row['Start Month']) & (dane['Start Day'] >= row['Start Day']))) &
                    (dane['End Day'].notna())
                    ]

            if len(relevant_data) < 4:
                relevant_data = dane[
                    (dane['Disaster Type'] == row['Disaster Type']) &
                    ((dane['Start Month'] > row['Start Month']) |
                     ((dane['Start Month'] == row['Start Month']) & (dane['Start Day'] >= row['Start Day']))) &
                    (dane['End Day'].notna())
                    ]

        # Obliczamy medianę 'End Day' z przefiltrowanych danych
        median_end_day = relevant_data['End Day'].median()

        # jeśli za mało danych, bierzemy ostatni dzień miesiąca
        if len(relevant_data) < 4:
            if row['End Month'] == 2:
                median_end_day = 28
            elif row['End Month'] in [1, 3, 5, 7, 8, 10, 12]:
                median_end_day = 31
            elif row['End Month'] in [4, 6, 9, 11]:
                median_end_day = 30

        # Uzupełniamy 'Start Day', jeśli wyznaczono medianę
        if pd.notna(median_end_day):
            row['End Day'] = int(median_end_day)

    return row

# Zastosowanie funkcji tylko do przefiltrowanych wierszy
updated_data = filtered_data_missing_end_day.apply(fill_end_day_for_row, axis=1)

# Aktualizacja oryginalnych danych
dane.update(updated_data)

dane = dane.dropna()

print("uzupełniono lub usunięto brakujące dane.")


def create_date_column(year, month, day):
    return pd.to_datetime(dict(year=year, month=month, day=day), errors='coerce')


dane['Start Date'] = create_date_column(dane['Start Year'], dane['Start Month'], dane['Start Day'])
dane['End Date'] = create_date_column(dane['End Year'], dane['End Month'], dane['End Day'])

dane = dane.drop(['Start Day', 'Start Month', 'Start Year', 'End Day', 'End Month', 'End Year'], axis=1)

#usuwamy nulle z dat
dane = dane.dropna(subset=['Start Date', 'End Date'])

szereg_czasowy_katastrofy = []

for _, row in dane.iterrows():
    # Generowanie zakresu dat od 'Start Date' do 'End Date'
    date_range = pd.date_range(start=row['Start Date'], end=row['End Date'])
    for date in date_range:
        szereg_czasowy_katastrofy.append({
            'Date': date,
            'Disaster Type': row['Disaster Type'],
            'Country': row['Country'],
            'Subregion': row['Subregion'],
            'Region': row['Region']
        })

# # Tworzenie nowego DataFrame
# szereg_czasowy_katastrofy_df = pd.DataFrame(szereg_czasowy_katastrofy)
# print('wykonano szereg czasowy')
#
# szereg_czasowy_katastrofy_df.head()
#
# countries = szereg_czasowy_katastrofy_df['Country'].unique()
#
# start_date = pd.Timestamp("2000-01-01")
# end_date = pd.Timestamp("2024-10-30")
# full_date_range = pd.date_range(start=start_date, end=end_date)
#
# complete_time_series = []
#
# for country in countries:
#     # Pełny zakres dat dla danego kraju
#     full_country_dates = pd.DataFrame({'Date': full_date_range, 'Country': country})
#
#     # Dołączenie informacji o katastrofach
#     country_disaster_data = szereg_czasowy_katastrofy_df[
#         szereg_czasowy_katastrofy_df['Country'] == country
#         ]
#
#     merged_data = full_country_dates.merge(
#         country_disaster_data,
#         on=['Date', 'Country'],
#         how='left'
#     )
#
#     # Uzupełnienie braków dla dat bez katastrof
#     merged_data['Subregion'] = merged_data['Subregion'].fillna(country_to_subregion.get(country, 'Unknown'))
#     merged_data['Region'] = merged_data['Region'].fillna(country_to_region.get(country, 'Unknown'))
#     merged_data['Disaster Type'] = merged_data['Disaster Type'].fillna('None')
#     merged_data['Days Since Start'] = merged_data['Days Since Start'].fillna(-1)
#     merged_data['Days Till End'] = merged_data['Days Till End'].fillna(-1)
#     merged_data['Duration'] = merged_data['Duration'].fillna(-1)
#
#     complete_time_series.append(merged_data)
# print('uzupełniono szereg czasowy datami bez katastrof')
#
#
# # Łączenie wszystkich danych w jeden DataFrame
# complete_time_series_df = pd.concat(complete_time_series, ignore_index=True)
#
# # Konwersja kolumny Date na datetime
# complete_time_series_df['Date'] = pd.to_datetime(complete_time_series_df['Date'])
#
# # Dodanie dnia roku
# complete_time_series_df['Day_of_Year'] = complete_time_series_df['Date'].dt.dayofyear
#
# # Dodanie sinusa i cosinusa dnia roku
# complete_time_series_df['Sin_Day_of_Year'] = np.sin(2 * np.pi * complete_time_series_df['Day_of_Year'] / 365)
# complete_time_series_df['Cos_Day_of_Year'] = np.cos(2 * np.pi * complete_time_series_df['Day_of_Year'] / 365)
#
#
#
#
#
#
# regions = complete_time_series_df['Region'].unique()
# region_dataframes = {region: complete_time_series_df[complete_time_series_df['Region'] == region] for region in regions}
#
# # Wyświetlenie nazw regionów i liczby wierszy dla każdego z nich
# for region, df in region_dataframes.items():
#     print(f"Region: {region}, Liczba wierszy: {len(df)}")
#
# europe_df = complete_time_series_df[complete_time_series_df['Region'] == 'Europe'].copy()
# europe_df.head()
#
# europe_df['Is_disaster'] = (europe_df['Disaster Type'] != 'None').astype(int)
# complete_df = pd.get_dummies(europe_df, columns=['Disaster Type'], drop_first=False)
#
#
# # complete_time_series_df['Is_disaster'] = (complete_time_series_df['Disaster Type'] != 'None').astype(int)
# # complete_df = pd.get_dummies(complete_time_series_df, columns=['Disaster Type'], drop_first=False)
#
# #musimy usunąć te kolumny, ponieważ wskazują na to, czy jest katastrofa czy nie.
# combined_df = complete_df.drop(['Duration', 'Days Since Start', 'Days Till End'], axis=1)
#
# combined_df = (
#     combined_df
#     .groupby(['Date', 'Country'], as_index=False)
#     .agg({
#         'Is_disaster': 'max',  # Zachowaj 1, jeśli w jakiejkolwiek katastrofie wystąpiło
#         **{col: 'max' for col in complete_df.columns if 'Disaster Type_' in col},
#         **{col: 'first' for col in complete_df.columns if col.startswith('Region')},
#         **{col: 'first' for col in complete_df.columns if col.startswith('Subregion')},
#         # Zachowaj stałe wartości dla Subregion i Region
#         **{col: 'first' for col in complete_df.columns if col.startswith('Country')}  # Zachowaj wartości dla krajów
#     })
# )
#
# print(combined_df.columns)
# combined_df = pd.get_dummies(combined_df, columns=['Region'], drop_first=False)
#
# print('kolumny po get_dummies regionów')
# print(combined_df.columns)
# combined_df = pd.get_dummies(combined_df, columns=['Subregion'], drop_first=False)
# print('kolumny po get_dummies subregionów')
# print(combined_df.columns)
# combined_df = pd.get_dummies(combined_df, columns=['Country'], drop_first=False)
# print('kolumny po get_dummies krajów')
# print(combined_df.columns)
#
# # Przygotowanie danych
# df_lstm = combined_df
# # df_lstm['Day_of_Year'] = pd.to_datetime(df_lstm['Date']).dt.dayofyear
#
# # Usuń kolumnę 'Date' i ustaw 'Is_disaster' jako cel
# features = df_lstm.drop(columns=['Date', 'Is_disaster'])
# target = df_lstm['Is_disaster']
#
# # Konwersja do NumPy
# features_matrix = features.values
# target_array = target.values
#
# # Parametr: długość sekwencji
# sequence_length = 365
#
# # Tworzenie danych dla LSTM
# grouped = complete_time_series_df.groupby('Country')
#
# X_lstm_updated, y_lstm_updated = [], []
# for i in range(len(features_matrix) - sequence_length):
#     X_lstm_updated.append(features_matrix[i:i + sequence_length])
#     y_lstm_updated.append(target_array[i + sequence_length])
# X_lstm_updated, y_lstm_updated = np.array(X_lstm_updated), np.array(y_lstm_updated)
#
# # Reshape danych wejściowych do formatu akceptowanego przez LSTM
# X_lstm_updated = X_lstm_updated.astype('float32')
# y_lstm_updated = y_lstm_updated.astype('float32')
#
# print(X_lstm_updated.dtype)  # Oczekiwany: float32
# print(y_lstm_updated.dtype)  # Oczekiwany: float32 lub int32
#
# # Oversampling klasy 1.0
# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(
#     X_lstm_updated.reshape(-1, X_lstm_updated.shape[1] * X_lstm_updated.shape[2]), y_lstm_updated
# )
# X_resampled = X_resampled.reshape(-1, X_lstm_updated.shape[1], X_lstm_updated.shape[2])
#
# # Podział danych na treningowe i testowe
# X_train_lstm_updated, X_test_lstm_updated, y_train_lstm_updated, y_test_lstm_updated = train_test_split(
#     X_resampled, y_resampled, test_size=0.3, random_state=42
# )
#
# # Ważenie klas
# class_weights = {
#     0: len(y_train_lstm_updated) / np.sum(y_train_lstm_updated == 0),
#     1: len(y_train_lstm_updated) / np.sum(y_train_lstm_updated == 1),
# }
#
# # Budowa ulepszonego modelu LSTM
# model_lstm_updated = Sequential()
#
# # Warstwa LSTM z Dropout i Batch Normalization
# model_lstm_updated.add(LSTM(100, activation='relu', return_sequences=True,
#                             input_shape=(X_train_lstm_updated.shape[1], X_train_lstm_updated.shape[2])))
# model_lstm_updated.add(Dropout(0.2))  # Dropout zapobiega przeuczeniu
# model_lstm_updated.add(BatchNormalization())  # Normalizacja dla stabilności uczenia
#
# # Druga warstwa LSTM
# model_lstm_updated.add(LSTM(50, activation='relu', return_sequences=False))
# model_lstm_updated.add(Dropout(0.2))  # Kolejny Dropout
#
# # Warstwa Dense z większą liczbą neuronów
# model_lstm_updated.add(Dense(20, activation='relu'))  # Warstwa ukryta dla większej złożoności
# model_lstm_updated.add(Dropout(0.2))  # Dropout na warstwie ukrytej
#
# # Warstwa wyjściowa
# model_lstm_updated.add(Dense(1, activation='sigmoid'))  # Aktywacja sigmoid dla klasyfikacji binarnej
#
# # Kompilacja modelu
# model_lstm_updated.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Trenowanie modelu
# model_lstm_updated.fit(
#     X_train_lstm_updated, y_train_lstm_updated,
#     epochs=10, batch_size=32,
#     validation_data=(X_test_lstm_updated, y_test_lstm_updated),
#     class_weight=class_weights
# )
#
# # Prognoza na danych testowych
# predictions_test = model_lstm_updated.predict(X_test_lstm_updated)
# predictions_binary_test = (predictions_test > 0.5).astype(int)
#
# # Ocena modelu
# print(classification_report(y_test_lstm_updated, predictions_binary_test))
#
# # AUC-ROC
# roc_auc = roc_auc_score(y_test_lstm_updated, predictions_test)
# print("AUC-ROC:", roc_auc)
#
# # Metryki regresji
# from sklearn.metrics import mean_squared_error, mean_absolute_error
#
# mse = mean_squared_error(y_test_lstm_updated, predictions_test)
# mae = mean_absolute_error(y_test_lstm_updated, predictions_test)
# rmse = mse ** 0.5
#
# print("MSE:", mse)
# print("MAE:", mae)
# print("RMSE:", rmse)
#
#
# def save_model_to_pkl(model, file_name='model.pkl'):
#     """
#     Zapisuje model LSTM do pliku .pkl i wyświetla komunikat po zapisaniu.
#
#     :param model: Model LSTM do zapisania
#     :param file_name: Nazwa pliku, w którym model zostanie zapisany (domyślnie 'model.pkl')
#     """
#     try:
#         # Zapisanie modelu do pliku .pkl
#         with open(file_name, 'wb') as file:
#             pickle.dump(model, file)
#
#         # Wyświetlenie komunikatu sukcesu
#         print(f"Model został zapisany w pliku '{file_name}'")
#     except Exception as e:
#         # Wyświetlenie błędu, jeśli coś pójdzie nie tak
#         print(f"Wystąpił błąd podczas zapisywania modelu: {e}")
#
#
# save_model_to_pkl(model_lstm_updated, 'model.pkl')