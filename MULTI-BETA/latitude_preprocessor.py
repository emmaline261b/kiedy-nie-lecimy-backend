class LatitudePreprocessor:
    COUNTRY_LATITUDES = {
        'Afghanistan': 33.93911,
        'Albania': 41.15333,
        'Algeria': 28.03389,
        'Andorra': 42.546245,
        'Angola': -11.202692,
        'Argentina': -38.416097,
        'Armenia': 40.0691,
        'Australia': -25.274398,
        'Austria': 47.516231,
        'Bangladesh': 23.684994,
        'Belgium': 50.503887,
        'Brazil': -14.235004,
        'Canada': 56.130366,
        'Chile': -35.675147,
        'China': 35.86166,
        'Colombia': 4.570868,
        'Cuba': 21.521757,
        'Denmark': 56.26392,
        'Ecuador': -1.831239,
        'Egypt': 26.820553,
        'Finland': 61.92411,
        'France': 46.603354,
        'Germany': 51.165691,
        'India': 20.593684,
        'Indonesia': -0.789275,
        'Italy': 41.87194,
        'Japan': 36.204824,
        'Kenya': -1.292066,
        'Malaysia': 4.210484,
        'Mexico': 23.634501,
        'Netherlands': 52.132633,
        'Nigeria': 9.081999,
        'Norway': 60.472024,
        'Peru': -9.189967,
        'Philippines': 12.879721,
        'Russia': 61.52401,
        'South Africa': -30.559482,
        'Spain': 40.463667,
        'Sweden': 60.128161,
        'Switzerland': 46.818188,
        'Thailand': 15.870032,
        'Turkey': 38.963745,
        'United Kingdom': 55.378051,
        'United States': 37.09024,
        'Vietnam': 14.058324,
        'Zimbabwe': -19.015438,
    }

    @staticmethod
    def get_latitude(country):
        """
        Get the latitude for a given country. If the country is not found, return None.
        """
        return LatitudePreprocessor.COUNTRY_LATITUDES.get(country, None)
