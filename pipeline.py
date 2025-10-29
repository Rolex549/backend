import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from preprocess import clean_text

def train_and_save():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(clean_text)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    model = MultinomialNB()
    model.fit(X, y)

    with open('saved_model/spam_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

if __name__ == '__main__':
    train_and_save()
