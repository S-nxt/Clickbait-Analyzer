# Clickbait Analyzer / Credibility Analyzer Project History

Last updated: 2026-09-25

## 1. Project purpose

This project is a Python-based article credibility checker designed to classify whether a piece of news is likely fake or real. Its core idea is to analyze the text of an article, clean it, convert it into TF-IDF features, and run multiple machine learning models to estimate the likelihood that the article is clickbait or misinformation.

The application accepts either:
- a raw article snippet or headline typed by the user
- a live URL to an article, which is then scraped and processed

It then outputs a probability score for each model, along with a simple FAKE/REAL label.

---

## 2. High-level summary

This is a multi-model news credibility system built around a desktop interface using PyQt5. The project combines:
- classical machine learning models (Logistic Regression and Random Forest)
- a PyTorch neural network classifier
- a lightweight scraping module for article extraction from the web
- a text-cleaning pipeline for preprocessing

The app is intended as a practical demo or prototype for fake-news detection, with emphasis on a user-friendly GUI rather than a production-grade misinformation detection system.

---

## 3. What the project does

### Main behavior
When the user enters text or a URL:
1. The app checks whether the input is a URL or plain text.
2. If it is a URL, `src/scraper.py` downloads the page and extracts the title and paragraph text.
3. The article text is cleaned using the same text preprocessing used during training.
4. The cleaned text is converted to TF-IDF vectors.
5. Three separate models predict whether the content is FAKE or REAL.
6. The results are displayed in a GUI text box.

### Model outputs
Each model returns:
- `prob_fake`: probability that the article is fake
- `label`: `FAKE` if probability > 0.5, else `REAL`

The displayed result is essentially a credibility or clickbait score.

---

## 4. Project structure

### Root-level files
- `main.py`  
  Application launcher. Creates the PyQt application and opens the main window.

- `train_models.py`  
  Trains the classical ML models and saves the TF-IDF vectorizer and model artifacts.

- `train_pytorch.py`  
  Trains the PyTorch neural network classifier using the same processed text data.

- `requirements.txt`  
  Lists the Python dependencies required to run and train the project.

- `HISTORY.md`  
  Project summary and handoff documentation.

### Source package
- `src/gui.py`  
  Contains the desktop interface logic for the app. It provides the main window and handles the analysis workflow.

- `src/predictor.py`  
  Loads all saved models and performs prediction on input text.

- `src/scraper.py`  
  Fetches a web page and extracts visible article text from the HTML.

- `src/text_cleaner.py`  
  Provides the shared cleaning logic used by both model training and prediction.

### Data folder
- `data/Fake_Real_News_Data.csv`  
  Main training dataset. Contains structured news article text and labels.

### Models folder
- `models/tfidf_vectorizer.pkl`  
  Saved TF-IDF vectorizer trained on cleaned article text.

- `models/logistic_model.pkl`  
  Trained logistic regression classifier.

- `models/random_forest_model.pkl`  
  Trained random forest classifier.

- `models/pytorch_model.pt`  
  Saved state dictionary for the PyTorch neural network classifier.

- `models/neural_net.pt`  
  Present in the project but not used by the current inference code. This appears to be a stale or duplicate artifact.

---

## 5. Dataset details

The dataset is loaded from:
- `data/Fake_Real_News_Data.csv`

The file contains at least these columns:
- `title`
- `text`
- `label`

The project combines the `title` and `text` into a single field called `full_text` before training:

```python
full_text = df['title'].fillna('') + " " + df['text'].fillna('')
```

This helps the model learn from headlines and article body together, which is useful for clickbait-style fake news detection.

### Label mapping
The training code converts labels as follows:
- `FAKE` -> `1`
- `REAL` -> `0`

This makes the output compatible with classifier models that expect binary target values.

---

## 6. Text preprocessing logic

The current cleaning logic is implemented in `train_models.py`:

```python
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)
```

This includes:
- lowercase conversion
- removal of punctuation and numbers
- removal of stop words
- joining remaining tokens into a clean string

The logic is intentionally simple and designed for a classic NLP pipeline rather than modern transformer models.

---

## 7. Model training workflow

### 7.1 TF-IDF vectorization
The code uses `TfidfVectorizer(max_features=5000)`.

This means:
- the cleaned article text is transformed into sparse numeric vectors
- only the top 5000 terms by frequency are kept
- the model receives a compressed representation of the article content

### 7.2 Logistic Regression
The project trains a `LogisticRegression` model to classify fake vs real content.

### 7.3 Random Forest
A `RandomForestClassifier(n_estimators=100, random_state=42)` is also trained on the same TF-IDF features.

### 7.4 PyTorch neural network
The PyTorch model is defined in `train_pytorch.py` and uses an architecture like this:

```python
class NewsClassifierNN(nn.Module):
    def __init__(self, input_dim):
        super(NewsClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
```

Forward pass:
- input -> hidden layer (128 units)
- ReLU activation
- dropout regularization
- output layer -> sigmoid

This makes it a binary classifier used for fake-news probability estimation.

### 7.5 Split and evaluation
The models use:
- train/test split with `test_size=0.2`
- random_state = 42

The training scripts evaluate performance with `accuracy_score` and print the accuracy to the console.

---

## 8. Runtime / inference pipeline

The inference path is handled by `src/predictor.py`.

### Model loading
At startup, the predictor loads:
- `models/tfidf_vectorizer.pkl`
- `models/logistic_model.pkl`
- `models/random_forest_model.pkl`
- `models/pytorch_model.pt`

### Predict function
The `predict(raw_text)` method does the following:
1. calls `clean_text(raw_text)`
2. transforms the text with the saved TF-IDF vectorizer
3. runs the logistic regression model
4. runs the random forest model
5. converts the TF-IDF features to a tensor and feeds the neural network
6. returns a dictionary of probabilities and labels

### Probability interpretation
The output from each model is a fake probability in the range `[0, 1]`:
- near `0` = more likely real
- near `1` = more likely fake

The app then converts that value to a percentage for display in the GUI.

---

## 9. Desktop app behavior

The GUI is implemented in `src/gui.py`.

### Interface elements
- text field for article URL or pasted text
- button labeled `Analyze Article`
- results window for the output

### Interaction flow
When the user clicks the button:
- the text is checked to see if it is a URL
- if URL, the scraper extracts article content
- the content is sent to the predictor
- predictions are formatted into a readable report

The result format is something like:

```text
--- ANALYSIS RESULTS ---

• Logistic Regression:
  Prediction: FAKE
  Clickbait / Fake Probability: 78.32%
```

---

## 10. Scraping behavior

The scraper in `src/scraper.py` uses `requests` and `BeautifulSoup` to:
- fetch the page HTML
- find the main `h1` tag as the title
- collect all `p` tags as article body text
- join the title and body into one text block

It checks whether the page content is too short and returns an error if the article is likely blocked or unreadable.

### Important note
This scraping code is intentionally simple and may fail or produce noisy output on sites with:
- JavaScript-rendered content
- anti-scraping protections
- unusual HTML structure
- pages with minimal text or poor semantic markup

---

## 11. How to run the app

### Install dependencies
```bash
pip install -r requirements.txt
```

### Launch the GUI app
```bash
python main.py
```

This starts the PyQt app and opens the credibility analyzer window.

---

## 12. How to retrain models

### Train the classical models
```bash
python train_models.py
```
This script:
- loads the dataset
- cleans the text
- fits a TF-IDF vectorizer
- trains logistic regression and random forest models
- saves them as `.pkl` files in `models/`

### Train the PyTorch network
```bash
python train_pytorch.py
```
This script:
- reuses the cleaned text workflow
- loads the saved TF-IDF vectorizer
- converts the data to tensors
- trains a neural network
- saves model weights to `models/pytorch_model.pt`

---

## 13. Current project status and known issues

This project is a working prototype and not a finished production solution. Several practical issues should be noted:

1. `src/text_cleaner.py` is now the shared cleaner used by training and the main app.  
   The actual cleaning function lives in `train_models.py`.

2. The model names are inconsistent.  
   The training script saves `pytorch_model.pt`, while a stale file `neural_net.pt` also exists in `models/`; the current app loads only `pytorch_model.pt`.

3. The project depends on a simplified dataset and rule-based text cleaning.  
   It is suitable for a demo or university project, not a robust real-world misinformation detector.

4. The scraper is basic.  
   It may fail on dynamic websites or pages that front-load content with JavaScript.

5. The models can only learn from the dataset they were trained on.  
   Misleading or domain-specific articles not similar to the training corpus may be misclassified.

---

## 14. Best maintenance points for future changes

This project is easy to extend if someone wants to improve it later.

### Likely update targets
- improve article scraping robustness
- add more dataset sources
- use more advanced preprocessing (lemmatization, n-grams, stop-word tuning)
- add model confidence thresholds and explainability output
- support multilingual text
- add a better UI with clickable results and source quality metrics
- add automated retraining and model versioning

### Most important code locations to adjust
- `src/gui.py` for UI behavior and result formatting
- `src/scraper.py` for content extraction logic
- `src/predictor.py` for model selection and probability handling
- `train_models.py` for feature engineering and classical model training
- `train_pytorch.py` for neural network architecture and training settings

---

## 15. Summary

This project is a desktop-based fake-news and clickbait detection tool built in Python. It combines three models—Logistic Regression, Random Forest, and a PyTorch neural network—on cleaned article text extracted from either user input or a scraped article URL. It is built as a practical AI project with a simple GUI, a basic NLP pipeline, and saved trained model artifacts.

Its purpose is educational and demonstrative: to classify news-like text as likely FAKE or REAL using classical machine learning and a neural network, while providing a clear user interface for testing and experimentation.

This documentation is intended to make the project easy to understand, hand off, extend, or debug without needing to read the whole codebase blindly.

Also update this file if anything is changed.