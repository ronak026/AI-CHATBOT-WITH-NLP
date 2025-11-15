# Task 3 — NLP Chatbot

A Python-based conversational chatbot that uses Natural Language Processing (NLP) techniques to understand user queries and provide relevant answers. The chatbot uses vectorization and cosine similarity to match user questions with a knowledge base, and includes intent detection for common conversational patterns.

## Features

- **Interactive Chat Interface**: Command-line chatbot that responds to user queries in real-time
- **NLP Processing**: Tokenization, lemmatization, and stopword removal for text preprocessing
- **Vector-based Similarity Matching**: Uses TF-IDF-like vectorization and cosine similarity to find the best matching answer
- **Intent Detection**: Recognizes greetings, goodbyes, and thanks for natural conversation flow
- **Extensible Knowledge Base**: Easy to add new Q/A pairs to expand the chatbot's knowledge
- **NLTK Integration**: Optional NLTK support with fallback to simple tokenization if NLTK is not available
- **Confidence Scoring**: Displays confidence scores for matched answers

## Project Structure

- `main.py`: Main entry point with the chatbot loop and answer matching logic
- `kb.py`: Knowledge base (Q/A pairs) and vocabulary/vector initialization
- `nlp_utils.py`: NLP utility functions (preprocessing, vectorization, similarity, intent detection)
- `requirements.txt`: Python dependencies

## Prerequisites

- Python 3.9+ recommended
- NLTK (optional but recommended for better NLP processing)

## Setup

1. Clone/download the project and open the `Task-3` folder.

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Download NLTK data resources:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Run

From the `Task-3` directory:

```bash
python main.py
```

The chatbot will start and prompt you with: `NLP Chatbot (type 'exit' or 'quit' to stop)`

Type your questions or messages. Examples:

- "Hello" → Greeting response
- "What is your name?" → Matches KB entry
- "What is Python?" → Matches KB entry
- "Thank you" → Thanks response
- "exit" or "quit" → Exits the chatbot

## How It Works

1. **Text Preprocessing**: User input is tokenized, lowercased, and optionally lemmatized with stopwords removed
2. **Vectorization**: Preprocessed tokens are converted to a vector representation based on the vocabulary
3. **Similarity Matching**: Cosine similarity is calculated between the user's query vector and all knowledge base vectors
4. **Answer Selection**: The best matching answer (if confidence ≥ 0.2) is returned
5. **Intent Detection**: Special intents (greetings, goodbyes, thanks) are detected before similarity matching

## Extending the Knowledge Base

To add new Q/A pairs, edit `kb.py` and add entries to the `KB` list:

```python
KB = [
    # ... existing entries ...
    ("your question here", "Your answer here"),
]
```

After adding entries, the vocabulary and vectors will be automatically rebuilt when you run the script.

## Customization

### Adjusting Confidence Threshold

In `main.py`, modify the threshold in `find_best_answer()`:

```python
if best_score >= 0.2:  # Change this value (0.0 to 1.0)
```

### Adding New Intents

In `nlp_utils.py`, add new intent patterns to the `detect_intent()` function:

```python
def detect_intent(text):
    t = text.lower()
    # Add your custom intent detection here
    # ...
```

### Using spaCy (Alternative)

To use spaCy instead of NLTK, uncomment the spaCy line in `requirements.txt` and install a model:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Then modify `nlp_utils.py` to use spaCy for tokenization and lemmatization.

## Notes & Troubleshooting

- **NLTK Not Available**: The chatbot will work with simple tokenization if NLTK is not installed, though results may be less accurate
- **Low Confidence Answers**: If the chatbot frequently returns "I'm not sure about that", try:
  - Adding more Q/A pairs to the knowledge base
  - Lowering the confidence threshold (currently 0.2)
  - Using more similar wording in your questions to the KB entries
- **NLTK Download Issues**: If you encounter errors downloading NLTK data, run the download command manually or check your internet connection
- **Empty Responses**: The chatbot ignores empty input and continues prompting

## Example Interactions

```
You: Hello
Chatbot: Hello! How can I help you today?

You: What is your name?
Chatbot: I'm an NLP Chatbot built with Python. You can call me Chatbot. (confidence=0.71)

You: What is machine learning?
Chatbot: Machine learning is a field of AI that uses statistical techniques to give computer systems the ability to learn from data. (confidence=0.65)

You: Thanks!
Chatbot: You're welcome!

You: exit
Chatbot: Goodbye — have a nice day!
```

## License

MIT (or as applicable to your project).
