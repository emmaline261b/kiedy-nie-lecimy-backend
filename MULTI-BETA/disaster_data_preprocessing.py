import pandas as pd

class DisasterDataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        print("Plik został wczytany.")

    def fill_missing_start_month(self):
        filtered_data = self.data[
            self.data['start month'].isnull() &
            self.data['end month'].notna()
        ]

        def fill_start_month_for_row(row):
            if pd.isna(row['start month']) and pd.notna(row['end month']):
                relevant_data = self.data[
                    (self.data['disaster_type'] == row['disaster_type']) &
                    (self.data['subregion'] == row['subregion']) &
                    (self.data['end month'] <= row['end month']) &
                    (self.data['start month'].notna())
                ]

                if len(relevant_data) < 3:
                    relevant_data = self.data[
                        (self.data['disaster_type'] == row['disaster_type']) &
                        (self.data['region'] == row['region']) &
                        (self.data['end month'] <= row['end month']) &
                        (self.data['start month'].notna())
                    ]

                median_start_month = relevant_data['start month'].median()
                if pd.notna(median_start_month):
                    row['start month'] = int(median_start_month)
            return row

        updated_data = filtered_data.apply(fill_start_month_for_row, axis=1)
        self.data.update(updated_data)

    def fill_missing_end_month(self):
        filtered_data = self.data[
            self.data['end month'].isnull() &
            self.data['start month'].notna()
        ]

        def fill_end_month_for_row(row):
            if pd.isna(row['end month']) and pd.notna(row['start month']):
                relevant_data = self.data[
                    (self.data['disaster_type'] == row['disaster_type']) &
                    (self.data['subregion'] == row['subregion']) &
                    (self.data['start month'] >= row['start month']) &
                    (self.data['end month'].notna())
                ]

                if len(relevant_data) < 4:
                    relevant_data = self.data[
                        (self.data['disaster_type'] == row['disaster_type']) &
                        (self.data['region'] == row['region']) &
                        (self.data['start month'] >= row['start month']) &
                        (self.data['end month'].notna())
                    ]

                median_end_month = relevant_data['end month'].median()
                if pd.notna(median_end_month):
                    row['end month'] = int(median_end_month)
            return row

        updated_data = filtered_data.apply(fill_end_month_for_row, axis=1)
        self.data.update(updated_data)

    def fill_missing_days(self):
        def fill_day(row, col_to_fill, col_reference):
            relevant_data = self.data[
                (self.data['disaster_type'] == row['disaster_type']) &
                (self.data['subregion'] == row['subregion']) &
                (self.data[col_reference] <= row[col_reference]) &
                (self.data[col_to_fill].notna())
            ]

            median_day = relevant_data[col_to_fill].median()
            if pd.notna(median_day):
                row[col_to_fill] = int(median_day)
            return row

        for col_to_fill, col_reference in [('start day', 'end month'), ('end day', 'start month')]:
            filtered_data = self.data[
                self.data[col_to_fill].isna() &
                self.data[col_reference].notna()
            ]

            updated_data = filtered_data.apply(
                lambda row: fill_day(row, col_to_fill, col_reference), axis=1
            )
            self.data.update(updated_data)

    def create_date_columns(self):
        def create_date_column(year, month, day):
            return pd.to_datetime(dict(year=year, month=month, day=day), errors='coerce')

        self.data['start date'] = create_date_column(
            self.data['start year'], self.data['start month'], self.data['start day']
        )
        self.data['end date'] = create_date_column(
            self.data['end year'], self.data['end month'], self.data['end day']
        )

        self.data = self.data.drop([
            'start day', 'start month', 'start year', 'end day', 'end month', 'end year'
        ], axis=1)

    def preprocess(self):
        self.fill_missing_start_month()
        self.fill_missing_end_month()
        self.fill_missing_days()
        self.create_date_columns()
        self.data = self.data.dropna(subset=['start date', 'end date'])
        print("Preprocessing zakończony. Dane są gotowe.")

    def save_to_csv(self):
        output_path = self.filepath.replace('.csv', '_preprocessed.csv')
        self.data.to_csv(output_path, index=False)
        print(f"Przetworzone dane zapisano do pliku: {output_path}")

    def get_data(self):
        return self.data
