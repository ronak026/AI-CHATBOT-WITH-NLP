import re
from math import sqrt

# Optional NLTK usage
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False


def simple_tokenize(text):
    tokens = re.split(r"\W+", text.lower())
    return [t for t in tokens if t]


if NLTK_AVAILABLE:
    try:
        lemmatizer = WordNetLemmatizer()
        STOPWORDS = set(stopwords.words('english'))
    except Exception:
        lemmatizer = None
        STOPWORDS = set()
else:
    lemmatizer = None
    STOPWORDS = set()


def normalize_tokens(tokens):
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if lemmatizer is not None:
            try:
                out.append(lemmatizer.lemmatize(t))
            except Exception:
                out.append(t)
        else:
            out.append(t)
    return out


def preprocess(text):
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text.lower())
        except Exception:
            tokens = simple_tokenize(text)
    else:
        tokens = simple_tokenize(text)
    return normalize_tokens(tokens)


def build_vocab(documents):
    vocab = {}
    for doc in documents:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def to_vector(tokens, vocab):
    vec = [0] * len(vocab)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1
    return vec


def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon"]
GOODBYES = ["bye", "goodbye", "see you", "exit", "quit"]
THANKS = ["thanks", "thank you", "thx"]


def detect_intent(text):
    t = text.lower()
    for g in GREETINGS:
        if g in t:
            return 'greeting'
    for g in GOODBYES:
        if g in t:
            return 'goodbye'
    for g in THANKS:
        if g in t:
            return 'thanks'
    return None
