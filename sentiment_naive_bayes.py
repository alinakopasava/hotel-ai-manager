import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# dane z data.py
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

train_df['Review_cleaned'] = train_df['Review_cleaned'].fillna("")
test_df['Review_cleaned'] = test_df['Review_cleaned'].fillna("")

# zamiana tekstu na liczby przez TF-IDF 
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(train_df['Review_cleaned'])
X_test_vec = vectorizer.transform(test_df['Review_cleaned'])

# model Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, train_df['Sentiment'])

# ocena modelu
y_pred = nb_model.predict(X_test_vec)

accuracy = accuracy_score(test_df['Sentiment'], y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(test_df['Sentiment'], y_pred, average='binary')

print("\n--- WYNIKI DLA SENTYMENTU (NAIVE BAYES) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nPełny raport klasyfikacji:")
print(classification_report(test_df['Sentiment'], y_pred))

# zapisuje model i wektoryzator jako opcję zapasową
joblib.dump(nb_model, 'sentiment_naive_bayes.pkl')
joblib.dump(vectorizer, 'sentiment_vectorizer.pkl')
