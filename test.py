import joblib
import pandas as pd
import numpy as np

# ładowanie modeli i wektoryzatorów
try:
    # XGBoost
    xgb_model = joblib.load('xgb_category_model.pkl')
    vectorizer_xgb = joblib.load('vectorizer_advanced.pkl')
    label_encoder = joblib.load('label_encoder.pkl') 

    # Regresja Logistyczna
    lr_model = joblib.load('logistic_regression_model.pkl')
    vectorizer_lr = joblib.load('vectorizer_baseline.pkl') 
    

except FileNotFoundError as e:
    print(f"Błąd: brakuje plików modeli ({e})")
    exit()

def test_single_review(text):
    print(f"\nRecenzja do sprawdzenia: '{text}'")
    print("-" * 50)
    
    # XGBoost 
    X_xgb = vectorizer_xgb.transform([text])
    pred_idx = xgb_model.predict(X_xgb)[0]
    # zmienia numeru klasy z powrotem na nazwę
    cat_xgb = label_encoder.inverse_transform([pred_idx])[0]
    
    # regresja
    X_lr = vectorizer_lr.transform([text])
    cat_lr = lr_model.predict(X_lr)[0]
    
    # porównanie obu wyników
    print(f"[XGBoost]:      {cat_xgb}")
    print(f"[LogReg]:      {cat_lr}")
    
    if cat_xgb == cat_lr:
        print("Modele są zgodne.")
    else:
        print("Rożbieżność")

# pętla żeby móc wpisywać przykłady 
print("\nWpisz recenzje (lub 'exit' aby wyjść):")

while True:
    user_input = input("\nTreść recenzji: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    if not user_input.strip():
        continue
        
    test_single_review(user_input)