import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already available
nltk.download('stopwords', quiet=True)

# Load standard NLTK English stopwords
default_stopwords = set(stopwords.words('english'))

# List of critical negation words to PRESERVE in text analysis
NEGATION_WORDS = {
    'no', 'not', 'nor', 'neither', 'never', 'none', 
    'doesnt', 'isnt', 'wasnt', 'arent', 'werent', 
    'wouldnt', 'couldnt', 'shouldnt', 'cant', 'cannot', 'dont'
}

# Remove negation words from the stopword removal list
CUSTOM_STOPWORDS = default_stopwords - NEGATION_WORDS


def clean_text(text):
    """
    Cleans raw text while preserving negations essential for context and sentiment.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase text
    text = text.lower()
    
    # 2. Keep only letters and spaces (removes special symbols)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Filter out stopwords EXCEPT preserved negation words
    words = text.split()
    cleaned_words = [word for word in words if word not in CUSTOM_STOPWORDS]
    
    return " ".join(cleaned_words)