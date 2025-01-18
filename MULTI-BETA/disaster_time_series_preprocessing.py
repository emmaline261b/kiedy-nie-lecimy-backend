import pandas as pd
import numpy as np
import os

from hemisphere_preprocessor import HemispherePreprocessor
from season_preprocessor import SeasonPreprocessor
from latitude_preprocessor import LatitudePreprocessor


class DisasterTimeSeriesPreprocessor:
    def __init__(self, processed_data):
        self.data = processed_data
        self.hemisphere = HemispherePreprocessor()
        self.season = SeasonPreprocessor()
        self.latitude = LatitudePreprocessor()

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

        # Dodanie cech tygodnia roku i miesiąca
        complete_time_series['week_of_year'] = complete_time_series['date'].dt.isocalendar().week
        complete_time_series['month_of_year'] = complete_time_series['date'].dt.month

        return complete_time_series

    def add_hemisphere(self, complete_time_series):
        # Przypisanie półkuli na podstawie kraju
        complete_time_series['hemisphere'] = complete_time_series['country'].apply(
            self.hemisphere.determine_hemisphere
        )
        return complete_time_series

    def add_season(self, complete_time_series):
        # Przypisanie pory roku na podstawie daty i półkuli
        complete_time_series['season'] = complete_time_series.apply(
            lambda row: self.season.determine_season(row['date'], row['hemisphere']), axis=1
        )
        return complete_time_series

    def add_latitude_features(self, complete_time_series):
        # Dodanie szerokości geograficznej jako sinus i cosinus
        complete_time_series['latitude'] = complete_time_series['country'].apply(
            self.latitude.get_latitude
        )

        # Dodanie sinus i cosinus szerokości geograficznej
        complete_time_series['sin_latitude'] = np.sin(
            2 * np.pi * complete_time_series['latitude']
        )
        complete_time_series['cos_latitude'] = np.cos(
            2 * np.pi * complete_time_series['latitude']
        )

        return complete_time_series


    def save_time_series(self, complete_time_series):
        # Sortowanie danych wg kolumny Country, a następnie Date
        complete_time_series = complete_time_series.sort_values(by=["country", "date"])

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

        print("Dodawanie półkuli...")
        complete_time_series = self.add_hemisphere(complete_time_series)

        print("Dodawanie pory roku...")
        complete_time_series = self.add_season(complete_time_series)

        print("Dodawanie szerokości geograficznej...")
        complete_time_series = self.add_latitude_features(complete_time_series)


        print("Zapisywanie danych...")
        self.save_time_series(complete_time_series)
        print("Przetwarzanie zakończone.")

        return complete_time_series
