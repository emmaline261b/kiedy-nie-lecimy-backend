from datetime import date

from disaster_data_preprocessing import DisasterDataPreprocessor
from disaster_predictor import DisasterPredictor
from disaster_time_series_preprocessing import DisasterTimeSeriesPreprocessor
from multi_class_classification_neural_network_modelling import MultiClassClassificationModel

import pickle

# # Inicjalizacja i wczytanie danych
# preprocessor = DisasterDataPreprocessor('resources/EMDAT-disaster-dataset-2000-2024-cleaned.csv')
# preprocessor.load_data()
#
# # Przetwarzanie danych
# preprocessor.preprocess()
#
# #zapisz do pliku na wszelki wypadek
# preprocessor.save_to_csv()
#
# # Pobranie przetworzonych danych
# processed_data = preprocessor.get_data()
#
# #====================================================
# #TWORZENIE SZEREGU CZASOWEGO
# #====================================================
#
# # Inicjalizacja preprocesora
# preprocessor = DisasterTimeSeriesPreprocessor(processed_data)
#
# # Przetwarzanie danych
# processed_time_series = preprocessor.preprocess()
#
# #====================================================
# #MODELOWANIE
# #====================================================
#
# # Użycie klasy
# model = MultiClassClassificationModel(processed_time_series)
# encoder = model.run()

#====================================================
#PRZYKŁADOWE UŻYCIE
#====================================================

# Wczytaj encoder z pliku
with open("../output/country_encoder.pkl", "rb") as file:
    country_encoder = pickle.load(file)

# Inicjalizacja predyktora
predictor = DisasterPredictor()
future_date = date(2025, 2, 12)
results = predictor.predict("Poland", future_date)
print("Top 3 przewidywane katastrofy:", results)