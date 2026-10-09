# import pandas as pd
# import joblib
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from src.text_cleaner import clean_text

# print("1. Loading Data & TF-IDF Vectorizer...")
# df = pd.read_csv("data/Fake_Real_News_Data.csv")
# df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
# df['clean_text'] = df['full_text'].apply(clean_text)
# df['target'] = df['label'].map({'FAKE': 1, 'REAL': 0})

# # Load the saved 10,000-feature vectorizer
# vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
# X = vectorizer.transform(df['clean_text']).toarray()
# y = df['target'].values

# # FIX 1: 70/30 Train/Test split with stratification
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# X_train_t = torch.tensor(X_train, dtype=torch.float32)
# y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
# X_test_t = torch.tensor(X_test, dtype=torch.float32)
# y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)


# class NewsClassifierNN(nn.Module):
#     def __init__(self, input_dim):
#         super(NewsClassifierNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(128, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)
#         return self.sigmoid(out)


# # FIX 2: Dynamically pass input_dim (10,000) instead of hardcoding 5000
# model = NewsClassifierNN(input_dim=X_train.shape[1])
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# print(f"2. Training PyTorch Neural Network (Input Dim: {X_train.shape[1]})...")
# for epoch in range(3):
#     model.train()
#     total_loss = 0
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         predictions = model(batch_X)
#         loss = criterion(predictions, batch_y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}/3 Complete. Avg Loss: {total_loss / len(train_loader):.4f}")

# # FIX 3: Save to models/pytorch_model.pth
# torch.save(model.state_dict(), "models/pytorch_model.pth")
# print("\nPyTorch model trained and saved to models/pytorch_model.pth!")

import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.text_cleaner import clean_text

# 1. Model Definition (Safe to import anywhere)
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

def train_pytorch_model():
    # 2. Load dataset & remove duplicate rows
    df = pd.read_csv("data/Fake_Real_News_Data.csv")
    df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    
    initial_count = len(df)
    df.drop_duplicates(subset=['full_text'], inplace=True)
    print(f"Removed {initial_count - len(df)} duplicate rows. Cleaned dataset size: {len(df)}")

    # Preprocessing & Vectorization
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    X = vectorizer.transform(df['cleaned_text']).toarray()
    y = (df['label'] == 'FAKE').astype(int).values

    # Stratified 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Model, Loss, Optimizer
    model = NewsClassifierNN(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    print("\n--- Training PyTorch Neural Network ---")
    model.train()
    epochs = 5
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    # 3. FIX: Test Set Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_labels = (test_preds.numpy() > 0.5).astype(int)
        acc = accuracy_score(y_test, test_labels)

    print(f"\nPyTorch Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, test_labels))

    # Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pytorch_model.pth")
    print("Saved PyTorch model to models/pytorch_model.pth")

# FIX: Wrap execution in main check so importing NewsClassifierNN does NOT trigger retraining
if __name__ == "__main__":
    train_pytorch_model()