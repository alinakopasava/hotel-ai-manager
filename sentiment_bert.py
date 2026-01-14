import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# dane przygotowane w data.py
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# sprawdzamy czy nie ma pustych pól w tekście żeby program się nie wywalił
train_df['Review_cleaned'] = train_df['Review_cleaned'].fillna("")
test_df['Review_cleaned'] = test_df['Review_cleaned'].fillna("")

# standardowy tokenizer dla modelu BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# gotowe paczki danych do treningu sentymentu
train_dataset = ReviewDataset(
    train_df['Review_cleaned'].tolist(), 
    train_df['Sentiment'].tolist(), 
    tokenizer, 
    max_length=128
)
test_dataset = ReviewDataset(
    test_df['Review_cleaned'].tolist(), 
    test_df['Sentiment'].tolist(), 
    tokenizer, 
    max_length=128
)

# model BERT z dwoma etykietami (pozytywny i negatywny)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# liczy statystyki do raportu 
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    accuracy = accuracy_score(p.label_ids, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ustawienia treningu
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=100,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# trening
trainer.train()

# ostateczne wyniki
print("\n--- WYNIKI DLA SENTYMENTU (BERT) ---")
results = trainer.evaluate()
print("\n:")
print(results)

# zapis modelu
trainer.save_model('./model')
tokenizer.save_pretrained('./model')