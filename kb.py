from nlp_utils import preprocess, build_vocab, to_vector

# Knowledge Base (expandable)
KB = [
    ("what is your name", "I'm an NLP Chatbot built with Python. You can call me Chatbot."),
    ("how are you", "I'm a program, so I don't have feelings, but I'm ready to help you!"),
    ("what can you do", "I can answer simple questions, demonstrate NLP techniques, and be extended with more data."),
    ("how to install nltk", "Run: pip install nltk and then use nltk.download() to get resources."),
    ("thank you", "You're welcome! Happy to help."),
    ("what is python", "Python is a popular high-level programming language for many purposes."),
    ("tell me a joke", "Why did the programmer quit his job? Because he didn't get arrays."),
    ("what is ai", "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems."),
    ("define machine learning", "Machine learning is a field of AI that uses statistical techniques to give computer systems the ability to learn from data."),
    ("what is natural language processing", "Natural Language Processing (NLP) is a field of AI focused on the interaction between computers and human (natural) languages."),
    ("how to use this chatbot", "Just type your question. Add new Q/A pairs in kb.py to expand my knowledge."),
]

# Pre-tokenize KB
KB_tokens = [preprocess(q) for q, _ in KB]

# Build vocabulary
VOCAB = build_vocab(KB_tokens)

# Vectorize KB
KB_vectors = [to_vector(tokens, VOCAB) for tokens in KB_tokens]
