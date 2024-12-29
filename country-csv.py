import pandas as pd
import os

wczytany_plik = pd.read_csv('resources/katastrofy_do_2024-12-28.csv')
print('wczytano plik')

dane=wczytany_plik

# Filtrujemy dane
filtered_data = dane[
  dane['Start Month'].isnull() &
  dane['End Month'].notna()]

def fill_start_month_for_row(row):
  if pd.isna(row['Start Month']) and pd.notna(row['End Month']):
    # Filtrujemy dane dla tego samego typu katastrofy i subregionu
    relevant_data = dane[
      (dane['Disaster_Type'] == row['Disaster_Type']) &
      (dane['Subregion'] == row['Subregion']) &
      (dane['End Month'] <= row['End Month']) &  # Warunek End Month
      (dane['Start Month'].notna())  # Tylko wiersze z istniejącym Start Month
      ]

    if len(relevant_data) < 3:
      relevant_data = dane[
        (dane['Disaster_Type'] == row['Disaster_Type']) &
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
      (dane['Disaster_Type'] == row['Disaster_Type']) &
      (dane['Subregion'] == row['Subregion']) &
      (dane['Start Month'] >= row['Start Month']) &  # Warunek Start Month
      (dane['End Month'].notna())  # Tylko wiersze z istniejącym End Month
      ]


    if len(relevant_data) < 4:
      relevant_data = dane[
        (dane['Disaster_Type'] == row['Disaster_Type']) &
        (dane['Region'] == row['Region']) &
        (dane['Start Month'] >= row['Start Month']) &  # Warunek Start Month
        (dane['End Month'].notna())  # Tylko wiersze z istniejącym End Month
      ]

    if len(relevant_data) < 4:
      relevant_data = dane[
        (dane['Disaster_Type'] == row['Disaster_Type']) &
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
dane.update(updated_data_end_month)

# Filtrujemy dane z brakującym 'Start Day' i znanym 'End Month'
filtered_data_missing_day = dane[
  dane['Start Day'].isna() &
  dane['End Month'].notna()
]

def fill_start_day_for_row(row):
  relevant_data = None

  if pd.isna(row['Start Day']) and pd.notna(row['End Month']):

    if pd.isna(row['End Day']): # with no end day
      relevant_data = dane[
        (dane['Disaster_Type'] == row['Disaster_Type']) &
        (dane['Subregion'] == row['Subregion']) &
        (dane['End Month'] <= row['End Month']) &
        (dane['Start Day'].notna())
        ]

      if len(relevant_data) < 4:
        relevant_data = dane[
          (dane['Disaster_Type'] == row['Disaster_Type']) &
          (dane['Region'] == row['Region']) &
          (dane['End Month'] <= row['End Month']) &
          (dane['Start Day'].notna())
        ]

      if len(relevant_data) < 4:
        relevant_data = dane[
          (dane['Disaster_Type'] == row['Disaster_Type']) &
          (dane['End Month'] <= row['End Month']) &
          (dane['Start Day'].notna())
        ]


    elif pd.notna(row['End Day']): # with end day
      relevant_data = dane[
        (dane['Disaster_Type'] == row['Disaster_Type']) &
        (dane['Subregion'] == row['Subregion']) &
        ((dane['End Month'] < row['End Month']) |
          ((dane['End Month'] == row['End Month']) & (dane['End Day'] <= row['End Day']))) &
        (dane['Start Day'].notna())
      ]

      if len(relevant_data) < 4:
        relevant_data = dane[
          (dane['Disaster_Type'] == row['Disaster_Type']) &
          (dane['Region'] == row['Region']) &
          ((dane['End Month'] < row['End Month']) |
          ((dane['End Month'] == row['End Month']) & (dane['End Day'] < row['End Day']))) &
          (dane['Start Day'].notna())
        ]

      if len(relevant_data) < 4:
        relevant_data = dane[
          (dane['Disaster_Type'] == row['Disaster_Type']) &
          ((dane['End Month'] < row['End Month']) |
          ((dane['End Month'] == row['End Month']) & (dane['End Day'] < row['End Day']))) &
          (dane['Start Day'].notna())
        ]

    # Obliczamy medianę 'Start Day' z przefiltrowanych danych
    median_start_day = relevant_data['Start Day'].median()

    #jeśli za mało danych, bierzemy ostatni dzień miesiąca
    if len(relevant_data) < 4:
      median_start_day = 1

    # Uzupełniamy 'Start Day', jeśli wyznaczono medianę
    if pd.notna(median_start_day):
        row['Start Day'] = int(median_start_day)

  return row

# Zastosowanie funkcji tylko do przefiltrowanych wierszy
updated_data = filtered_data_missing_day.apply(fill_start_day_for_row, axis=1)
dane.update(updated_data)

# Filtrujemy dane z brakującym 'End Day' i znanym 'Start Month'
filtered_data_missing_end_day = dane[
  dane['End Day'].isna() &
  dane['Start Month'].notna()
]

def fill_end_day_for_row(row):
  relevant_data = None

  if pd.isna(row['End Day']) and pd.notna(row['Start Month']):

    if pd.isna(row['Start Day']): # with no start day
      relevant_data = dane[
        (dane['Disaster_Type'] == row['Disaster_Type']) &
        (dane['Subregion'] == row['Subregion']) &
        (dane['Start Month'] >= row['Start Month']) &
        (dane['End Day'].notna())
        ]

      if len(relevant_data) < 4:
        relevant_data = dane[
          (dane['Disaster_Type'] == row['Disaster_Type']) &
          (dane['Region'] == row['Region']) &
          (dane['Start Month'] >= row['Start Month']) &
          (dane['End Day'].notna())
        ]

      if len(relevant_data) < 4:
        relevant_data = dane[
          (dane['Disaster_Type'] == row['Disaster_Type']) &
          (dane['Start Month'] >= row['Start Month']) &
          (dane['End Day'].notna())
        ]


    elif pd.notna(row['Start Day']): # with start day
      relevant_data = dane[
        (dane['Disaster_Type'] == row['Disaster_Type']) &
        (dane['Subregion'] == row['Subregion']) &
        ((dane['Start Month'] > row['Start Month']) |
          ((dane['Start Month'] == row['Start Month']) & (dane['Start Day'] >= row['Start Day']))) &
        (dane['End Day'].notna())
      ]

      if len(relevant_data) < 4:
        relevant_data = dane[
          (dane['Disaster_Type'] == row['Disaster_Type']) &
          (dane['Region'] == row['Region']) &
          ((dane['Start Month'] > row['Start Month']) |
            ((dane['Start Month'] == row['Start Month']) & (dane['Start Day'] >= row['Start Day']))) &
          (dane['End Day'].notna())
        ]

      if len(relevant_data) < 4:
        relevant_data = dane[
          (dane['Disaster_Type'] == row['Disaster_Type']) &
          ((dane['Start Month'] > row['Start Month']) |
            ((dane['Start Month'] == row['Start Month']) & (dane['Start Day'] >= row['Start Day']))) &
          (dane['End Day'].notna())
        ]


    # Obliczamy medianę 'End Day' z przefiltrowanych danych
    median_end_day = relevant_data['End Day'].median()

    #jeśli za mało danych, bierzemy ostatni dzień miesiąca
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
dane.update(updated_data)

print("Liczba brakujących wartości w każdej kolumnie:")
print(dane.isnull().sum())
dane = dane.dropna()


#==========FEATURE ENGINEERING =========

def create_date_column(year, month, day):
    return pd.to_datetime(dict(year=year, month=month, day=day), errors='coerce')

dane.loc[:, 'Start Date'] = create_date_column(dane['Start Year'], dane['Start Month'], dane['Start Day'])
dane.loc[:, 'End Date'] = create_date_column(dane['End Year'], dane['End Month'], dane['End Day'])

dane = dane.drop(['Start Day', 'Start Month','Start Year','End Day','End Month','End Year'], axis=1)

print(dane[['Start Date', 'End Date']].isna().sum())
dane = dane.dropna(subset=['Start Date', 'End Date'])


#======zapisujemy osobne csv dla każdego kraju ======
# Tworzymy folder na pliki CSV
output_folder = "datasets_by_country"
os.makedirs(output_folder, exist_ok=True)

# Grupujemy dane według kraju
grouped = dane.groupby('Country')

for country, group in grouped:
    # Tworzymy nazwę pliku na podstawie nazwy kraju
    file_name = f"dataset_{country.replace(' ', '_').replace('/', '_')}.csv"
    file_path = os.path.join(output_folder, file_name)

    # Zapisujemy dane dla danego kraju do pliku CSV
    group.to_csv(file_path, index=False)

print(f"Dane zostały podzielone i zapisane w folderze '{output_folder}'.")

