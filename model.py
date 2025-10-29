import pickle
import re

# 🔹 Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# 🔹 Load model and vectorizer from saved pickle file
def load_model():
    with open('saved_model/spam_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['vectorizer']

# 🔹 Predict spam or ham
def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)[0]
    return 'spam' if result == 1 else 'green flag'
