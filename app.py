from flask import Flask, request, jsonify
from flask_cors import CORS
from model import load_model, predict

app = Flask(__name__)
CORS(app)

model, vectorizer = load_model()

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    text = data.get('text', '')
    prediction = predict(text, model, vectorizer)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
