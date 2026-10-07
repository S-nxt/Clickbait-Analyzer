import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

_NEGATION_WORDS = {
    'no', 'not', 'nor', 'neither', 'never', 'none',
    'cannot', 'cant', 'couldnt', 'doesnt', 'dont', 'isnt',
    'shouldnt', 'wasnt', 'wont', 'wouldnt',
}
_STOPWORDS = set(stopwords.words('english')) - _NEGATION_WORDS


def clean_text(text: object) -> str:
    """Normalize article text using the same rules used during training."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = [word for word in text.split() if word not in _STOPWORDS]
    return " ".join(words)