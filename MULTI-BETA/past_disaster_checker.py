import pandas as pd

class PastDisasterChecker:
    def __init__(self, time_series_path="output/complete_time_series.csv"):
        self.time_series_path = time_series_path
        self.time_series_data = self._load_time_series()
        print(self.time_series_data.head())

    def _load_time_series(self):
        """
        Wczytuje dane z pliku zawierającego szereg czasowy katastrof.
        """
        try:
            return pd.read_csv(self.time_series_path, parse_dates=['date'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Nie znaleziono pliku {self.time_series_path}")

    def check_disaster(self, country, date):
        """
        Sprawdza, czy w danym kraju i dniu wystąpiła katastrofa.

        Args:
            country (str): Nazwa kraju.
            date (datetime.date): Data do sprawdzenia.

        Returns:
            dict: Sformatowany wynik z informacjami o katastrofach.
        """
        # Filtruj dane dla kraju i daty
        filtered = self.time_series_data[(self.time_series_data['country'] == country) & (self.time_series_data['date'] == pd.Timestamp(date))]

        if filtered.empty:
            return {"response": [{"none": "100.00%"}]}  # Brak katastrof dla tej daty
        else:
            # Zbierz unikalne typy katastrof i ich udział procentowy
            disaster_counts = filtered['disaster_type'].value_counts(normalize=True) * 100
            return {
                "response": [
                    {
                        "disaster": disaster,
                        "probability": f"{prob:.2f}%"
                    }
                    for disaster, prob in disaster_counts.items()
                ]
            }
