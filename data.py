import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    words = [w for w in words if len(w) > 1]
    return " ".join(words)

def load_and_augment(base_df, file_name):
    # czy plik z dodatkowymi danymi w ogole jest w folderze
    if os.path.exists(file_name):
        aug_df = pd.read_csv(file_name)
        # czyszczenie tekstu jak w glownej bazie
        aug_df['Review_cleaned'] = aug_df['Review'].apply(clean_text)
        print(f"Dodano {len(aug_df)} rekordów z {file_name}")
        return pd.concat([base_df, aug_df], ignore_index=True)
    return base_df

# wczytanie danych z tripadvisor i czyszczenie 
df_raw = pd.read_csv('tripadvisor_hotel_reviews.csv')
df_raw.dropna(subset=['Review', 'Rating'], inplace=True)
df_raw['Review_cleaned'] = df_raw['Review'].apply(clean_text)
# robie sentyment: 4 i 5 to pozytywne (1), reszta negatywne (0)
df_raw['Sentiment'] = df_raw['Rating'].apply(lambda x: 1 if x >= 4 else 0)

# dzielienie na zbior trenningowy i testowy
train_df, test_df = train_test_split(df_raw, test_size=0.2, random_state=42)

# wygenerowane dane z sarkazmem i lokalizacja do treningu
train_df = load_and_augment(train_df, 'sarcasm_augmentation.csv')
train_df = load_and_augment(train_df, 'location_augmentation.csv')

# usuniecie kategorii z testowych zeby model musial je sam zgadnac
if 'category' in test_df.columns: 
    test_df.drop(columns=['category'], inplace=True)

# zapisanie gotowych plików do csv
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)