class HemispherePreprocessor:
    NORTH_HEMISPHERE_COUNTRIES = {
        'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
        'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Canada', 'China',
        'Croatia', 'Cuba', 'Cyprus', 'Czechia', 'Denmark', 'Dominican Republic', 'Egypt', 'El Salvador', 'Estonia',
        'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Grenada', 'Guatemala', 'Hungary', 'Iceland', 'Iran',
        'Iraq', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Latvia',
        'Lebanon', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Mexico', 'Moldova', 'Monaco', 'Mongolia',
        'Montenegro', 'Morocco', 'Nepal', 'Netherlands', 'Nicaragua', 'North Korea', 'North Macedonia', 'Norway',
        'Pakistan', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Saint Kitts and Nevis', 'Saint Lucia',
        'Saint Vincent and the Grenadines', 'San Marino', 'Saudi Arabia', 'Serbia', 'Slovakia', 'Slovenia',
        'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Trinidad and Tobago',
        'Tunisia', 'Turkey', 'Turkmenistan', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States',
        'Uzbekistan', 'Vatican City', 'Yemen'
    }

    SOUTH_HEMISPHERE_COUNTRIES = {
        'Angola', 'Argentina', 'Australia', 'Bolivia', 'Botswana', 'Burundi', 'Chile', 'Eswatini', 'Fiji', 'Lesotho',
        'Madagascar', 'Malawi', 'Namibia', 'Nauru', 'New Zealand', 'Paraguay', 'Peru', 'Rwanda', 'Samoa',
        'Solomon Islands', 'South Africa', 'Tanzania', 'Timor-Leste', 'Tonga', 'Tuvalu', 'Uganda', 'Uruguay',
        'Vanuatu', 'Zambia', 'Zimbabwe'
    }

    TROPICAL_COUNTRIES = {
        'Belize', 'Benin', 'Cambodia', 'Central African Republic', 'Colombia', 'Congo', 'Costa Rica',
        'Democratic Republic of the Congo', 'Dominica', 'Ecuador', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
        'Guinea-Bissau', 'Haiti', 'India', 'Indonesia', 'Kenya', 'Kiribati', 'Liberia', 'Maldives', 'Malaysia',
        'Mauritania', 'Micronesia', 'Nigeria', 'Panama', 'Papua New Guinea', 'Philippines',
        'Saint Vincent and the Grenadines', 'Senegal', 'Seychelles', 'Sierra Leone', 'Singapore', 'Somalia',
        'South Sudan', 'Sri Lanka', 'Suriname', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Vietnam'
    }

    @staticmethod
    def determine_hemisphere(country):
        if country in HemispherePreprocessor.NORTH_HEMISPHERE_COUNTRIES:
            return 'north'
        elif country in HemispherePreprocessor.SOUTH_HEMISPHERE_COUNTRIES:
            return 'south'
        elif country in HemispherePreprocessor.TROPICAL_COUNTRIES:
            return 'tropical'
        else:
            return 'unknown'
