import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import os

# słownik 
def expert_labeling(text):
    t = str(text).lower()
    
    if any(w in t for w in ["metro station", "subway", "airport shuttle", "bus stop", "central location", "neighborhood", "city center", "ocean view", "mountain view", "walking distance to"]):
        return "Location & View"
    
    if any(w in t for w in ["dirty", "mold", "stain", "dust", "smell", "filthy", "disgusting", "cockroach", "unclean"]):
        return "Cleanliness"
    
    if any(w in t for w in ["breakfast buffet", "scrambled", "omelette", "coffee maker", "tasty food", "delicious dinner", "restaurant menu"]):
        return "Food & Breakfast"
    
    if any(w in t for w in ["expensive", "overpriced", "refund", "receipt", "worth the money", "bill", "surcharge", "pricey"]):
        return "Price & Value"
    
    if any(w in t for w in ["wifi signal", "air conditioning", "swimming pool", "gym equipment", "fridge", "shower head", "pillow", "mattress"]):
        return "Amenities & Room"
    
    if any(w in t for w in ["receptionist", "manager", "helpful staff", "rude staff", "unprofessional", "ignored", "polite", "apology"]):
        return "Service & Management"

    return "Other"

# sprawdzam czy dane sa przygotowane
if not os.path.exists('train_data.csv') or not os.path.exists('test_data.csv'):
    print("Błąd: brakuje plików CSV")
    exit()

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# usuwamy kategorię Other
test_df['category'] = test_df['Review_cleaned'].apply(expert_labeling)

train_cat = train_df[train_df['category'] != "Other"].dropna(subset=['category']).copy()
test_cat = test_df[test_df['category'] != "Other"].dropna(subset=['category']).copy()

# trenowanie
vectorizer_cat = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer_cat.fit_transform(train_cat['Review_cleaned'])
X_test_vec = vectorizer_cat.transform(test_cat['Review_cleaned'])

# balansuje klasy sztucznie, zeby model nie faworyzowal najwiekszych grup
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_vec, train_cat['category'])

# regresja logistyczna (podejście testowe)
model_cat = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model_cat.fit(X_train_res, y_train_res)

y_pred_cat = model_cat.predict(X_test_vec)
print("\n--- WYNIKI DLA KATEGORII (LOGISTIC REGRESSION) ---")
print(classification_report(test_cat['category'], y_pred_cat))

# zapis prostych modeli do porównania
joblib.dump(model_cat, 'logistic_regression_model.pkl')
joblib.dump(vectorizer_cat, 'vectorizer_baseline.pkl')
