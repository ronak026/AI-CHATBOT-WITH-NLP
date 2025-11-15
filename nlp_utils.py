"""
NLP utility functions for text preprocessing, vectorization, and intent detection.

This module provides functions for tokenization, lemmatization, vectorization,
cosine similarity calculation, and intent detection. It supports both NLTK-based
and simple fallback implementations.
"""

import re
import warnings
from math import sqrt

# Optional NLTK usage
NLTK_AVAILABLE = False
lemmatizer = None
STOPWORDS = set()

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
except Exception as e:
    warnings.warn(f"NLTK import failed: {e}. Using fallback tokenization.", UserWarning)
    NLTK_AVAILABLE = False

# Initialize NLTK components if available
if NLTK_AVAILABLE:
    try:
        lemmatizer = WordNetLemmatizer()
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        warnings.warn(
            "NLTK data not found. Please run: "
            "python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\"",
            UserWarning,
        )
        lemmatizer = None
        STOPWORDS = set()
    except Exception as e:
        warnings.warn(
            f"NLTK initialization failed: {e}. Using fallback methods.", UserWarning
        )
        lemmatizer = None
        STOPWORDS = set()


def simple_tokenize(text):
    """
    Simple tokenization using regex (fallback when NLTK is not available).

    Args:
        text: Input text string

    Returns:
        List of lowercase tokens (non-word characters removed)
    """
    if not text or not isinstance(text, str):
        return []
    tokens = re.split(r"\W+", text.lower())
    return [t for t in tokens if t]


def normalize_tokens(tokens):
    """
    Normalize tokens by removing stopwords and applying lemmatization.
    If all tokens would be filtered as stopwords, keeps at least one token
    to ensure short queries can still match.

    Args:
        tokens: List of token strings

    Returns:
        List of normalized tokens (stopwords removed, lemmatized if available)
    """
    if not tokens:
        return []
    out = []
    all_stopwords = []

    for t in tokens:
        if not t:
            continue
        if t in STOPWORDS:
            # Keep track of stopwords in case all tokens are stopwords
            all_stopwords.append(t)
            continue
        if lemmatizer is not None:
            try:
                out.append(lemmatizer.lemmatize(t))
            except Exception:
                out.append(t)
        else:
            out.append(t)

    # If all tokens were stopwords, keep at least one (the first one)
    # This ensures queries like "how are you" can still match
    if not out and all_stopwords:
        # Use the first stopword, lemmatized if possible
        first_token = all_stopwords[0]
        if lemmatizer is not None:
            try:
                out.append(lemmatizer.lemmatize(first_token))
            except Exception:
                out.append(first_token)
        else:
            out.append(first_token)

    return out


def preprocess(text):
    """
    Preprocess text: tokenize, normalize, and remove stopwords.

    Args:
        text: Input text string

    Returns:
        List of preprocessed tokens
    """
    if not text or not isinstance(text, str):
        return []

    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text.lower())
        except Exception:
            tokens = simple_tokenize(text)
    else:
        tokens = simple_tokenize(text)
    return normalize_tokens(tokens)


def build_vocab(documents):
    """
    Build a vocabulary dictionary from a list of tokenized documents.

    Args:
        documents: List of lists of tokens

    Returns:
        Dictionary mapping tokens to their index in the vocabulary
    """
    vocab = {}
    for doc in documents:
        if not doc:
            continue
        for token in doc:
            if token and token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def to_vector(tokens, vocab):
    """
    Convert a list of tokens to a vector representation (bag of words).

    Args:
        tokens: List of token strings
        vocab: Vocabulary dictionary mapping tokens to indices

    Returns:
        List representing the vector (count of each token in vocabulary)
    """
    if not vocab:
        return []
    vec = [0] * len(vocab)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1
    return vec


def cosine_sim(a, b):
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector (list of numbers)
        b: Second vector (list of numbers)

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# Intent detection patterns
GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
GOODBYES = ["bye", "goodbye", "see you", "exit", "quit", "farewell"]
THANKS = ["thanks", "thank you", "thx", "thank", "appreciate"]


def detect_intent(text):
    """
    Detect user intent from text (greeting, goodbye, thanks).
    Uses word boundary matching to avoid false positives from substrings.

    Args:
        text: Input text string

    Returns:
        Intent string ('greeting', 'goodbye', 'thanks') or None
    """
    if not text or not isinstance(text, str):
        return None

    t = text.lower().strip()
    if not t:
        return None

    # Split text into words for more accurate matching
    words = set(re.split(r"\W+", t))
    words.discard("")  # Remove empty strings

    # Check for greetings (using word boundaries)
    for g in GREETINGS:
        # For multi-word phrases, check if all words are present
        if " " in g:
            g_words = set(g.split())
            if g_words.issubset(words) or g in t:
                return "greeting"
        # For single words, check exact word match
        elif g in words:
            return "greeting"

    # Check for goodbyes
    for g in GOODBYES:
        if " " in g:
            g_words = set(g.split())
            if g_words.issubset(words) or g in t:
                return "goodbye"
        elif g in words:
            return "goodbye"

    # Check for thanks
    for g in THANKS:
        if " " in g:
            g_words = set(g.split())
            if g_words.issubset(words) or g in t:
                return "thanks"
        elif g in words:
            return "thanks"

    return None
