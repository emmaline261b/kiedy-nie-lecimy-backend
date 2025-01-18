import pandas as pd
import numpy as np
import os

class DisasterTimeSeriesPreprocessor:
    def __init__(self, processed_data):
        self.data = processed_data

    def generate_time_series(self):
        # Konwersja dat na typ datetime
        self.data['start date'] = pd.to_datetime(self.data['start date'])
        self.data['end date'] = pd.to_datetime(self.data['end date'])

        # Generowanie szeregu czasowego dla każdej katastrofy
        time_series_data = []
        for _, row in self.data.iterrows():
            date_range = pd.date_range(start=row['start date'], end=row['end date'])
            for date in date_range:
                time_series_data.append({
                    'date': date,
                    'country': row['country'],
                    'subregion': row['subregion'],
                    'region': row['region'],
                    'disaster_type': row['disaster_type']
                })

        # Tworzenie DataFrame ze szeregiem czasowym
        time_series_df = pd.DataFrame(time_series_data)
        return time_series_df

    def fill_missing_dates(self, time_series_df):
        # Pełny zakres dat
        start_date = pd.Timestamp("2000-01-01")
        end_date = pd.Timestamp("2024-12-31")
        full_date_range = pd.date_range(start=start_date, end=end_date)

        # Uzupełnianie brakujących dat dla każdego kraju
        complete_time_series = []
        for country in time_series_df['country'].unique():
            country_data = time_series_df[time_series_df['country'] == country]
            full_date_df = pd.DataFrame({'date': full_date_range})
            merged_data = full_date_df.merge(country_data, on='date', how='left')
            merged_data['country'] = country
            merged_data['disaster_type'] = merged_data['disaster_type'].fillna('None')
            merged_data['subregion'] = merged_data['subregion'].ffill().bfill()
            merged_data['region'] = merged_data['region'].ffill().bfill()
            complete_time_series.append(merged_data)

        # Łączenie danych dla wszystkich krajów
        return pd.concat(complete_time_series, ignore_index=True)

    def add_features(self, complete_time_series):
        # Dodanie cech sinus i cosinus dla dnia w roku
        complete_time_series['day_of_year'] = complete_time_series['date'].dt.dayofyear
        complete_time_series['sin_day_of_year'] = np.sin(2 * np.pi * complete_time_series['day_of_year'] / 365)
        complete_time_series['cos_day_of_year'] = np.cos(2 * np.pi * complete_time_series['day_of_year'] / 365)

        return complete_time_series

    def save_time_series(self, complete_time_series):
        # Zapisywanie całego szeregu czasowego do jednego pliku CSV
        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)

        file_path = os.path.join(output_folder, "complete_time_series.csv")
        complete_time_series.to_csv(file_path, index=False)
        print(f"Pełny szereg czasowy zapisano do pliku: {file_path}")

    def preprocess(self):
        print("Generowanie szeregu czasowego...")
        time_series_df = self.generate_time_series()

        print("Uzupełnianie brakujących dat...")
        complete_time_series = self.fill_missing_dates(time_series_df)

        print("Dodawanie cech...")
        complete_time_series = self.add_features(complete_time_series)

        print("Zapisywanie danych...")
        self.save_time_series(complete_time_series)
        print("Przetwarzanie zakończone.")

        return complete_time_series




