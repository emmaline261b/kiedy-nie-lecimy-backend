import pickle
import numpy as np
import pandas as pd
from datetime import date

class DisasterPredictor:
    def __init__(self, model_path="output/model.pkl", country_encoder_path="output/country_encoder.pkl", disaster_encoder_path="output/disaster_encoder.pkl"):
        # Wczytaj model z pliku
        self.model = self.load_model(model_path)
        self.country_encoder = self.load_encoder(country_encoder_path)
        self.disaster_encoder = self.load_encoder(disaster_encoder_path)

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                print(f"Model wczytano z: {model_path}")
                return model
        except FileNotFoundError:
            print(f"Nie znaleziono modelu w lokalizacji: {model_path}")
            raise

    def load_encoder(self, encoder_path):
        try:
            with open(encoder_path, 'rb') as file:
                encoder = pickle.load(file)
                print(f"Encoder wczytano z: {encoder_path}")
                return encoder
        except FileNotFoundError:
            print(f"Nie znaleziono encodera w lokalizacji: {encoder_path}")
            raise

    def predict(self, country, prediction_date):
        # Przetwarzanie daty na cechy sinusoidalne
        day_of_year = prediction_date.timetuple().tm_yday
        sin_day_of_year = np.sin(2 * np.pi * day_of_year / 365)
        cos_day_of_year = np.cos(2 * np.pi * day_of_year / 365)

        # Zakodowanie kraju
        country_vector = self.country_encoder.transform([[country]]).flatten()
        country_columns = self.country_encoder.get_feature_names_out()

        # Przygotowanie wektora wejściowego
        input_vector = {col.lower(): val for col, val in zip(country_columns, country_vector)}
        input_vector.update({
            'day_of_year_scaled': day_of_year,
            'sin_day_of_year_scaled': sin_day_of_year,
            'cos_day_of_year_scaled': cos_day_of_year
        })

        # Wczytanie oryginalnych nazw kolumn modelu
        with open("../output/model_columns.pkl", "rb") as file:
            model_columns = pickle.load(file)

        # Dodanie brakujących kolumn
        for col in model_columns:
            if col not in input_vector:
                input_vector[col] = 0

        # Usunięcie nadmiarowych kolumn
        input_vector = {col: input_vector[col] for col in model_columns}

        # Konwersja do DataFrame
        input_df = pd.DataFrame([input_vector])

        # Przewidywanie za pomocą modelu
        probabilities = self.model.predict(input_df, verbose=0)

        # Znalezienie top 3 wyników
        top_indices = np.argsort(probabilities[0])[-3:][::-1]
        disaster_labels = self.disaster_encoder.categories_[0]

        # Konwersja wyników na bardziej czytelny format
        top_disasters = [
            {
                "disaster": disaster_labels[i],
                "probability": f"{probabilities[0][i] * 100:.2f}%"  # Formatowanie jako procenty
            }
            for i in top_indices
        ]

        return {"response": top_disasters}