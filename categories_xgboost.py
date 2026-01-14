import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

# lista słów które wyrzucamy bo są w co drugiej recenzji i nic nie wnoszą
custom_stops = [
    'hotel', 'room', 'stay', 'guest', 'stayed', 'did', 'just', 'nice', 
    'good', 'great', 'really', 'bit', 'night', 'day', 'hotels', 'rooms', 
    'place', 'got', 'went', 'came', 'told', 'said'
]
my_stop_words = list(ENGLISH_STOP_WORDS.union(custom_stops))

# sprawdzam pliki i usuwam klase Other
if not os.path.exists('train_data.csv') or not os.path.exists('test_data.csv'):
    print("Błąd: brakuje plików CSV")
    exit()

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

train_df = train_df[train_df['category'] != "Other"].dropna(subset=['category', 'Review_cleaned'])
test_df = test_df[test_df['category'] != "Other"].dropna(subset=['category', 'Review_cleaned'])

# wyrównujemy liczbe przykładów dla każdej kategorii (max 800)
class_counts = train_df['category'].value_counts()
min_size = class_counts.min()
target_size = min(min_size, 800) 

train_balanced = pd.DataFrame()
for cat in train_df['category'].unique():
    df_cat = train_df[train_df['category'] == cat]
    df_resampled = resample(df_cat, 
                            replace=False, 
                            n_samples=target_size, 
                            random_state=42)
    train_balanced = pd.concat([train_balanced, df_resampled])

print("\n=== ROZKŁAD KLAS PO BALANSOWANIU ===")
print(train_balanced['category'].value_counts())

# zamiana tekstu na liczby - n-gramy 1-3
vectorizer = TfidfVectorizer(
    stop_words=my_stop_words,
    ngram_range=(1, 3),
    max_features=10000,
    min_df=2,
    max_df=0.7,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_balanced['Review_cleaned'])
X_test = vectorizer.transform(test_df['Review_cleaned'])

# XGBoost potrzebuje etykiet jako numery a nie nazwy
le = LabelEncoder()
y_train = le.fit_transform(train_balanced['category'])
y_test = le.transform(test_df['category'])

# trening
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    objective='multi:softprob',
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# wyniki i macierz pomyłek 
y_pred = model.predict(X_test)
print("\n--- WYNIKI DLA KATEGORII (XGBOOST) ---")
print(f"Ogólne Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))



plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=le.classes_, 
            yticklabels=le.classes_)
plt.title('Macierz Pomyłek - XGBoost (6 Klas)')
plt.ylabel('Prawdziwa klasa')
plt.xlabel('Przewidziana klasa (Model)')
plt.show()

# eksport do plików zeby odpalić w streamlicie 
joblib.dump(model, 'xgb_category_model.pkl')
joblib.dump(vectorizer, 'vectorizer_advanced.pkl')
joblib.dump(le, 'label_encoder.pkl')