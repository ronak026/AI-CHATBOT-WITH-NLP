"""
NLP Chatbot - Main Entry Point

A conversational chatbot that uses NLP techniques to match user queries
with a knowledge base using vector similarity and intent detection.
"""

from kb import KB, KB_tokens, KB_vectors, VOCAB
from nlp_utils import preprocess, to_vector, cosine_sim, detect_intent


def find_best_answer(user_text, confidence_threshold=0.2):
    """
    Find the best matching answer from the knowledge base for user input.

    Args:
        user_text: User's input text
        confidence_threshold: Minimum similarity score to return an answer (default: 0.2)

    Returns:
        Tuple of (answer_string, confidence_score) or (None, confidence_score)
    """
    if not user_text or not isinstance(user_text, str):
        return None, 0.0

    tokens = preprocess(user_text)
    if not tokens:
        return None, 0.0

    vec = to_vector(tokens, VOCAB)
    if not vec or not KB_vectors:
        return None, 0.0

    best_idx = None
    best_score = 0.0

    for i, kb_vec in enumerate(KB_vectors):
        s = cosine_sim(vec, kb_vec)
        if s > best_score:
            best_score = s
            best_idx = i

    if best_score >= confidence_threshold and best_idx is not None:
        return KB[best_idx][1], best_score
    else:
        return None, best_score


def main_loop():
    """
    Main interactive loop for the chatbot.
    Handles user input, intent detection, and answer retrieval.
    """
    print("=" * 60)
    print("NLP Chatbot")
    print("Type 'exit' or 'quit' to stop")
    print("=" * 60)
    print()

    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nChatbot: Goodbye!")
            break

        if not user.strip():
            continue

        user_lower = user.lower().strip()
        if user_lower in ("exit", "quit"):
            print("Chatbot: Goodbye — have a nice day!")
            break

        # Intent detection
        intent = detect_intent(user)
        if intent == "greeting":
            print("Chatbot: Hello! How can I help you today?")
            continue
        elif intent == "goodbye":
            print("Chatbot: Goodbye!")
            break
        elif intent == "thanks":
            print("Chatbot: You're welcome!")
            continue

        # Find best matching answer
        answer, score = find_best_answer(user)
        if answer:
            print(f"Chatbot: {answer} (confidence={score:.2f})")
        else:
            print("Chatbot: I'm not sure about that. You can extend the KB in kb.py.")
        print()


if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check that all dependencies are installed correctly.")
        raise
