import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join([w for w in text.split() if len(w) > 1])

    # słownik kategorii 
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

# uzupelnienie kategorii w zbiorach do uczenia
for file in ['train_data.csv', 'test_data.csv']:
    df = pd.read_csv(file)
    
    # jesli kolumna juz jest (np. z augmentacji) to dopisuje tylko tam gdzie jest pusto
    if 'category' not in df.columns:
        df['category'] = df['Review_cleaned'].apply(expert_labeling)
    else:
        # tripAdvisor dostaje etykiety ze slownika, a dane z csv zostaja tak jak byly
        df['category'] = df['category'].fillna(df['Review_cleaned'].apply(expert_labeling))
    
    df.to_csv(file, index=False)