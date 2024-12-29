import pickle

import pandas as pd
import os

# lstm
import numpy as np


wczytany_plik = pd.read_csv('datasets_by_country/dataset_Afghanistan.csv')
print('wczytano plik')

dane = wczytany_plik.drop(columns=['Subregion', 'Region', 'Country'])
print('kolumny w datasecie: ', dane.columns)

# Konwersja dat na typ datetime
wczytany_plik['Start Date'] = pd.to_datetime(wczytany_plik['Start Date'])
wczytany_plik['End Date'] = pd.to_datetime(wczytany_plik['End Date'])

# Generowanie szeregu czasowego
szereg_czasowy_katastrofy = []

for _, row in dane.iterrows():
    # Generowanie zakresu dat od 'Start Date' do 'End Date'
    date_range = pd.date_range(start=row['Start Date'], end=row['End Date'])
    for date in date_range:
        szereg_czasowy_katastrofy.append({
            'Date': date,
            'Disaster_Type': row['Disaster_Type']
        })


# Tworzenie DataFrame ze szeregiem czasowym
szereg_czasowy_katastrofy_df = pd.DataFrame(szereg_czasowy_katastrofy)
print('Wykonano szereg czasowy')

# Pełny zakres dat
start_date = pd.Timestamp("2000-01-01")
end_date = pd.Timestamp("2024-12-31")
full_date_range = pd.date_range(start=start_date, end=end_date)

# Uzupełnianie brakujących dat
full_date_df = pd.DataFrame({'Date': full_date_range})
merged_data = full_date_df.merge(szereg_czasowy_katastrofy_df, on='Date', how='left')
merged_data['Disaster_Type'] = merged_data['Disaster_Type'].fillna('None')
print('Uzupełniono brakujące daty')

# Dodanie cech
merged_data['Day_of_Year'] = merged_data['Date'].dt.dayofyear
merged_data['Sin_Day_of_Year'] = np.sin(2 * np.pi * merged_data['Day_of_Year'] / 365)
merged_data['Cos_Day_of_Year'] = np.cos(2 * np.pi * merged_data['Day_of_Year'] / 365)

# Dodanie flagi dla katastrof
merged_data['Is_disaster'] = (merged_data['Disaster_Type'] != 'None').astype(int)

# One-hot encoding kolumny 'Disaster_Type'
complete_df = pd.get_dummies(merged_data, columns=['Disaster_Type'], drop_first=False)

# Zamiana wartości True/False na 0/1 dla kolumn logicznych
for col in complete_df.columns:
    if complete_df[col].dtype == 'bool':  # Sprawdzenie, czy kolumna jest typu logicznego
        complete_df[col] = complete_df[col].map({True: 1, False: 0})
print("Zamieniono wartości True/False na 0/1 we wszystkich kolumnach")


# Grupowanie danych według unikalnych dat
combined_df = (
    complete_df
    .groupby(['Date'], as_index=False)
    .agg({
        'Is_disaster': 'max',  # Zachowaj 1, jeśli jakakolwiek katastrofa wystąpiła tego dnia
        **{col: 'max' for col in complete_df.columns if 'Disaster_Type_' in col},  # Zachowaj 1 dla typów katastrof
        'Day_of_Year': 'first',  # Zachowaj pierwszą wartość dla Day_of_Year
        'Sin_Day_of_Year': 'first',  # Zachowaj pierwszą wartość dla sinusoidalnej cechy
        'Cos_Day_of_Year': 'first'   # Zachowaj pierwszą wartość dla cosinusoidalnej cechy
    })
)
print('Dane zgrupowane po unikalnych datach')

# Tworzymy folder na pliki CSV
output_folder = "time_series_by_country"
os.makedirs(output_folder, exist_ok=True)

file_name = 'time_series_Afghanistan.csv'
file_path = os.path.join(output_folder, file_name)
combined_df.to_csv(file_path, index=False)

print("Szereg czasowy zapisano do pliku 'szereg_czasowy.csv'")