from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
CORS(app)  # Umożliwia obsługę zapytań z Angulara

# Załaduj model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


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


if __name__ == '__main__':
    app.run(debug=True)
