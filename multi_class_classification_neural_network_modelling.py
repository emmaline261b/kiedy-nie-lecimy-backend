import pandas as pd
import numpy as np
from keras.src.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import os

class MultiClassClassificationModel:
    def __init__(self, processed_time_series):
        self.data = processed_time_series
        self.model = None
        self.early_stopping = None

    def create_early_stopping(self, monitor='val_loss', patience=10, restore_best_weights=True, verbose=1):
        return EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=verbose
        )

    def preprocess_data(self):
        # kodowanie one-hot dla disaster_type
        disaster_encoder = OneHotEncoder(sparse_output=False)
        disaster_types = disaster_encoder.fit_transform(self.data[['disaster_type']])
        disaster_labels = disaster_encoder.categories_[0]

        # dodanie kodowanych etykiet do danych
        disaster_df = pd.DataFrame(disaster_types, columns=[f"disaster_{label}" for label in disaster_labels])
        self.data = pd.concat([self.data, disaster_df], axis=1)

        # skalowanie cech liczbowych
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.data[['day_of_year', 'sin_day_of_year', 'cos_day_of_year']])
        scaled_df = pd.DataFrame(scaled_features, columns=['day_of_year_scaled', 'sin_day_of_year_scaled', 'cos_day_of_year_scaled'])

        # połączenie danych wstępnie przetworzonych
        self.data = pd.concat([self.data.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

        # kodowanie one-hot dla country
        country_encoder = OneHotEncoder(sparse_output=False)
        country_encoded = country_encoder.fit_transform(self.data[['country']])
        country_labels = country_encoder.categories_[0]

        # dodanie kodowanych krajów do danych
        country_df = pd.DataFrame(country_encoded, columns=[f"country_{label}" for label in country_labels])
        self.data = pd.concat([self.data, country_df], axis=1)

        # kodowanie regionów i subregionów
        region_encoder = OneHotEncoder(sparse_output=False)
        region_encoded = region_encoder.fit_transform(self.data[['region']])
        region_labels = region_encoder.categories_[0]

        subregion_encoder = OneHotEncoder(sparse_output=False)
        subregion_encoded = subregion_encoder.fit_transform(self.data[['subregion']])
        subregion_labels = subregion_encoder.categories_[0]

        # dodanie kodowanych regionów i subregionów do danych
        region_df = pd.DataFrame(region_encoded, columns=[f"region_{label}" for label in region_labels])
        subregion_df = pd.DataFrame(subregion_encoded, columns=[f"subregion_{label}" for label in subregion_labels])
        self.data = pd.concat([self.data, region_df, subregion_df], axis=1)

        # zapisywanie encoderów
        os.makedirs("output", exist_ok=True)
        with open("output/disaster_encoder.pkl", 'wb') as file:
            pickle.dump(disaster_encoder, file)
        print("disaster encoder zapisano do pliku: output/disaster_encoder.pkl")

        with open("output/country_encoder.pkl", 'wb') as file:
            pickle.dump(country_encoder, file)
        print("country encoder zapisano do pliku: output/country_encoder.pkl")

        with open("output/region_encoder.pkl", 'wb') as file:
            pickle.dump(region_encoder, file)
        print("region encoder zapisano do pliku: output/region_encoder.pkl")

        with open("output/subregion_encoder.pkl", 'wb') as file:
            pickle.dump(subregion_encoder, file)
        print("subregion encoder zapisano do pliku: output/subregion_encoder.pkl")

        # tworzenie x i y
        feature_columns = ['day_of_year_scaled', 'sin_day_of_year_scaled', 'cos_day_of_year_scaled'] + \
                          [f"country_{label}" for label in country_labels] + \
                          [f"region_{label}" for label in region_labels] + \
                          [f"subregion_{label}" for label in subregion_labels]

        x = self.data[feature_columns]
        y = disaster_types

        # podział danych na treningowe i walidacyjne
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # zapisanie kolumn modelu
        model_columns = list(x.columns)
        with open("output/model_columns.pkl", "wb") as file:
            pickle.dump(model_columns, file)
        print("kolumny modelu zapisano do pliku: output/model_columns.pkl")

        return x_train, x_test, y_train, y_test

    def build_model(self, input_shape, output_shape):

        model = Sequential()
        model.add(Dense(256, input_shape=(input_shape,), activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(0.3))
        model.add(Dense(output_shape, activation='softmax'))

        model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train_model(self, x_train, y_train, epochs=200, batch_size=32, callbacks=None):
        if callbacks is None:
            callbacks = [self.create_early_stopping()]

        self.model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=callbacks
        )


    def save_model(self, file_path="output/model.pkl"):
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"model zapisano do pliku: {file_path}")

    def run(self):
        print("przetwarzanie danych...")
        x_train, x_test, y_train, y_test = self.preprocess_data()

        print("budowanie modelu...")
        self.build_model(input_shape=x_train.shape[1], output_shape=y_train.shape[1])

        print("trenowanie modelu...")
        self.train_model(x_train, y_train)

        print("zapisywanie modelu...")
        self.save_model()

        print("model gotowy.")
