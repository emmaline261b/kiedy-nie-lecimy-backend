class SeasonPreprocessor:
    @staticmethod
    def determine_season(date, hemisphere):
        month = date.month
        if hemisphere == 'north':
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            elif month in [9, 10, 11]:
                return 'autumn'
        elif hemisphere == 'south':
            if month in [12, 1, 2]:
                return 'summer'
            elif month in [3, 4, 5]:
                return 'autumn'
            elif month in [6, 7, 8]:
                return 'winter'
            elif month in [9, 10, 11]:
                return 'spring'
        else:  # Tropikalne regiony
            return 'wet-season' if month in [11, 12, 1, 2, 3] else 'dry-season'
