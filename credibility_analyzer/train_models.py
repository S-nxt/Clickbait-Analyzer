import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.text_cleaner import clean_text

print("1. Loading Dataset...")
df = pd.read_csv("data/Fake_Real_News_Data.csv")

df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
print("2. Cleaning Text (this takes ~10 seconds)...")
df['clean_text'] = df['full_text'].apply(clean_text)

df['target'] = df['label'].map({'FAKE': 1, 'REAL': 0})

# CHANGED: test_size=0.3 (70% Train / 30% Test) + stratify to keep label balance
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['target'], 
    test_size=0.3, 
    random_state=42,
    stratify=df['target']
)

print(f"Dataset Split -> Training: {len(X_train)} samples | Testing: {len(X_test)} samples")

print("3. Extracting TF-IDF Features (with Bi-grams & Tri-grams)...")
# MOVED: Define vectorizer settings BEFORE fitting
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the fitted vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("4. Training Logistic Regression...")
log_model = LogisticRegression()
log_model.fit(X_train_tfidf, y_train)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_model.predict(X_test_tfidf)))
joblib.dump(log_model, "models/logistic_model.pkl")

print("5. Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(X_test_tfidf)))
joblib.dump(rf_model, "models/random_forest_model.pkl")

print("\nAll models trained and saved to models/ folder!")