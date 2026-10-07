import joblib
import torch
import torch.nn as nn
from src.text_cleaner import clean_text
from train_pytorch import NewsClassifierNN

class ArticlePredictor:
    def __init__(self):
        self.vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        
        self.logistic = joblib.load("models/logistic_model.pkl")
        self.random_forest = joblib.load("models/random_forest_model.pkl")
        
        self.pytorch_model = NewsClassifierNN(input_dim=5000)
        self.pytorch_model.load_state_dict(torch.load("models/pytorch_model.pt"))
        self.pytorch_model.eval()

    def predict(self, raw_text):
        cleaned = clean_text(raw_text)
        tfidf_features = self.vectorizer.transform([cleaned])
        
        log_prob = self.logistic.predict_proba(tfidf_features)[0][1] 
        
        rf_prob = self.random_forest.predict_proba(tfidf_features)[0][1] 
        
        tensor_features = torch.tensor(tfidf_features.toarray(), dtype=torch.float32)
        with torch.no_grad():
            pt_prob = self.pytorch_model(tensor_features).item()
            
        return {
            "Logistic Regression": {"prob_fake": log_prob, "label": "FAKE" if log_prob > 0.5 else "REAL"},
            "Random Forest": {"prob_fake": rf_prob, "label": "FAKE" if rf_prob > 0.5 else "REAL"},
            "PyTorch Neural Net": {"prob_fake": pt_prob, "label": "FAKE" if pt_prob > 0.5 else "REAL"},
        }

if __name__ == "__main__":
    predictor = ArticlePredictor()
    sample_headline = "BREAKING: Scientists discover aliens living in local supermarket!"
    results = predictor.predict(sample_headline)
    print("Test Prediction Output:", results)