from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
CORS(app)  # Umożliwia obsługę zapytań z Angulara

# Załaduj model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Backend działa poprawnie!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pobierz dane z żądania JSON
        data = request.json
        input_data = data.get('input_data', [])

        # Predykcja
        prediction = model.predict([input_data])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/checkCatastrophe', methods=['POST'])
def check_catastrophe():
    try:
        # Pobierz dane z żądania JSON
        data = request.json
        date = data.get('date')  # Data przesłana w żądaniu
        country = data.get('country')  # Nazwa kraju przesłana w żądaniu

        if not date or not country:
            return jsonify({'error': 'Missing "date" or "country" in request'}), 400

        # Logika obsługi danych (dla przykładu, prosty response)
        # Możesz dodać swoje przetwarzanie danych tutaj
        result = {
            'date': date,
            'country': country,
            'catastrophe_detected': False  # Możesz zastąpić odpowiednią logiką
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
