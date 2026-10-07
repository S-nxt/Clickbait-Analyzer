import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.text_cleaner import clean_text
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

print("1. Loading Data & TF-IDF Vectorizer...")
df = pd.read_csv("data/Fake_Real_News_Data.csv")
df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
df['clean_text'] = df['full_text'].apply(clean_text)
df['target'] = df['label'].map({'FAKE': 1, 'REAL': 0})

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X = vectorizer.transform(df['clean_text']).toarray()
y = df['target'].values

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Tokenize preserves negations, word order, and full sentence context natively
def tokenize_func(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

class NewsClassifierNN(nn.Module):
    def __init__(self, input_dim):
        super(NewsClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)

model = NewsClassifierNN(input_dim=5000)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("2. Training PyTorch Neural Network (3 Epochs)...")
for epoch in range(3):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/pytorch_model.pt")
print("\nPyTorch model trained and saved to models/pytorch_model.pt!")