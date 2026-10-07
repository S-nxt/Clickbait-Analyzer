# Clickbait & Credibility Analyzer

A Python desktop application that analyzes news article text and estimates whether it is likely to be fake or real. The application can analyze pasted text or fetch article text from a URL, then compares predictions from Logistic Regression, Random Forest, and a PyTorch neural network.

This is an educational prototype, not a definitive fact-checking service. Its predictions depend on the training dataset and should be treated as indicators rather than proof.

## Features

- Analyze pasted headlines or article text.
- Analyze an article URL using a lightweight web scraper.
- Compare three machine learning models.
- Display a FAKE/REAL label and fake-probability percentage for each model.
- Retrain the models using the included dataset.

## Requirements

- Windows, macOS, or Linux
- Python 3.9 or newer recommended
- Internet access for URL scraping and for the first NLTK stopwords download
- A working Python environment with the packages in `requirements.txt`

## Installation

Open a terminal in the project directory:

```text
credibility_analyzer/
```

Create and activate a virtual environment. On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Run the application

The repository already includes trained model files in `models/`. After installing dependencies, launch the graphical application with:

```bash
python main.py
```

On Windows, `py main.py` can also be used.

In the application:

1. Enter a complete URL beginning with `http://` or `https://`, or paste article text.
2. Select **Analyze Article**.
3. Review the prediction and fake-probability percentage for each model.

For URL analysis, the scraper extracts the page's first `h1` heading and all paragraph elements. Some websites may fail because of JavaScript rendering, anti-bot protection, login requirements, or unusual HTML structure.

## Retrain the models

Retraining is optional. The dataset is stored at `data/Fake_Real_News_Data.csv` and must contain `title`, `text`, and `label` columns. Labels are expected to be `FAKE` and `REAL`.

Run the classical machine learning training script first:

```bash
python train_models.py
```

This creates or replaces:

- `models/tfidf_vectorizer.pkl`
- `models/logistic_model.pkl`
- `models/random_forest_model.pkl`

Then train the PyTorch model:

```bash
python train_pytorch.py
```

This creates or replaces:

- `models/pytorch_model.pt`

The scripts use the same preprocessing and an 80/20 train/test split with `random_state=42`. `train_models.py` also downloads the NLTK English stopwords resource if it is not already installed.

After retraining, run `python main.py` again so the application loads the updated artifacts.

## How prediction works

1. The title and article body are combined during training.
2. Text is lowercased, stripped of non-letter characters, and filtered for English stopwords.
3. A TF-IDF vectorizer keeps up to 5,000 features.
4. The same vector representation is passed to all three classifiers.
5. A probability greater than `0.5` is labeled `FAKE`; otherwise it is labeled `REAL`.

The probability is a model confidence estimate for the fake class. It does not verify claims against external sources.

## Project structure

```text
credibility_analyzer/
├── main.py                    # Application entry point
├── train_models.py            # TF-IDF, Logistic Regression, Random Forest training
├── train_pytorch.py           # PyTorch neural network training
├── requirements.txt           # Python dependencies
├── HISTORY.md                 # Detailed project handoff and maintenance notes
├── README.md                  # Setup and usage guide
├── data/
│   ├── Fake_Real_News_Data.csv
│   └── news_dataset.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   ├── pytorch_model.pt
│   └── neural_net.pt          # Present but not used by current inference code
└── src/
	├── gui.py                 # PyQt5 user interface
	├── predictor.py           # Model loading and inference
	├── scraper.py             # URL fetching and article text extraction
	├── text_cleaner.py        # Shared text preprocessing for training and prediction
	└── __init__.py
```

## Troubleshooting

### `FileNotFoundError` for model files

Run commands from the project root, the directory containing `main.py`, and confirm that the required files exist in `models/`. If they do not, run both training scripts in the order shown above.

### NLTK stopwords error

Run:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

Then retry training.

### URL scraping fails

Try pasting the article text directly. The scraper only looks for an `h1` title and HTML paragraph tags and cannot reliably process every website.

### PyTorch model loading error

Make sure `models/pytorch_model.pt` was generated by the current `train_pytorch.py` and that the TF-IDF vectorizer still has 5,000 or fewer features matching the neural network input size.

For a more detailed technical description, see [HISTORY.md](HISTORY.md).
