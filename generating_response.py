import os
import torch
import joblib
import google.generativeai as genai
from dotenv import load_dotenv
import csv
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification

# klucz do Gemini z pliku .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Błąd: nie znaleziono GEMINI_API_KEY")

genai.configure(api_key=api_key)

gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# ścieżka do modelu BERT
model_path = './model'

try:
    # ładuje model BERT 
    tokenizer = BertTokenizer.from_pretrained(model_path)
    rev_model = BertForSequenceClassification.from_pretrained(model_path)
    rev_model.eval()

    # ładuje XGBoost 
    cat_model = joblib.load('xgb_category_model.pkl')
    vectorizer_adv = joblib.load('vectorizer_advanced.pkl')
    label_encoder = joblib.load('label_encoder.pkl') 
    

except Exception as e:
    print(f"Błąd podczas ładowania modeli: {e}")
    exit()

def export_to_log(review, sentiment, category):
    # zapisuje każdą analizę do pliku CSV 
    file_path = 'hotel_issues_log.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Review Content', 'Sentiment', 'Category'])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            review,
            sentiment,
            category
        ])

def generate_response_with_gemini(review, sentiment, category):
    prompt = f"""
    Context: You are a professional hotel manager.
    Task: Write a short, polite response to a guest review.
    
    Detected Sentiment: {sentiment}
    Review Category: {category}
    
    Instructions:
    - If the sentiment is negative, apologize sincerely for the issues regarding {category} and promise improvement.
    - If the sentiment is positive, thank the guest for their kind words about {category}.
    - Keep the response professional and concise (max 3 sentences).
    - Language: English.
    """
    
    response = gemini_model.generate_content(prompt)
    return response.text

    # główna funkcja która zbiera wszystko 
def generate_polite_response_with_models(review):

    
    # sentyment (BERT)
    encoding = tokenizer.encode_plus(
        review, 
        add_special_tokens=True, 
        max_length=128, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    with torch.no_grad():
        outputs = rev_model(input_ids, attention_mask=attention_mask)
        sentiment_prediction = torch.argmax(outputs.logits, dim=1).item()

    sentiment = "positive" if sentiment_prediction == 1 else "negative"

    # kategoria (XGBoost)
    review_vec = vectorizer_adv.transform([review])
    category_idx = cat_model.predict(review_vec)[0]

    # zamieniam numer kategorii na nazwę
    category = label_encoder.inverse_transform([category_idx])[0]

    # zapis wyniku do logów
    export_to_log(review, sentiment, category)

    # odpowiedź z Gemini
    response_text = generate_response_with_gemini(review, sentiment, category)

    return {
        "review": review,
        "sentiment": sentiment,
        "category": category,
        "generated_response": response_text
    }