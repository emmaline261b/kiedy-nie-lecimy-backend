from flask import Flask, request, jsonify
from flask_cors import CORS
from disaster_predictor import DisasterPredictor
from past_disaster_checker import PastDisasterChecker
from datetime import datetime

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
CORS(app)

# Inicjalizacja klas
predictor = DisasterPredictor()
checker = PastDisasterChecker()

@app.route('/')
def home():
    return "Backend działa poprawnie!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pobierz dane z żądania JSON
        data = request.json
        country = data.get('country')
        date_str = data.get('date')

        # Walidacja wejścia
        if not country or not date_str:
            return jsonify({'error': 'Missing "country" or "date" in request'}), 400

        # Konwersja daty
        try:
            prediction_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

        # Obsługa zakresów dat
        if prediction_date < datetime(2000, 1, 1).date():
            return jsonify({'error': 'Model does not support historical data before 2000-01-01.'}), 400
        elif prediction_date <= datetime(2024, 12, 31).date():
            # Sprawdź w klasie PastDisasterChecker
            response = checker.check_disaster(country, prediction_date)
        else:
            # Sprawdź w modelu DisasterPredictor
            response = predictor.predict(country, prediction_date)

        # Konwersja wartości numpy.float32 na float
        for entry in response["response"]:
            for key in entry:
                entry[key] = str(entry[key]) if isinstance(entry[key], float) else entry[key]

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
