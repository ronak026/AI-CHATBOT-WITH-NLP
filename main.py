from kb import KB, KB_tokens, KB_vectors, VOCAB
from nlp_utils import preprocess, to_vector, cosine_sim, detect_intent


def find_best_answer(user_text):
    tokens = preprocess(user_text)
    vec = to_vector(tokens, VOCAB)
    best_idx = None
    best_score = 0.0
    for i, kb_vec in enumerate(KB_vectors):
        s = cosine_sim(vec, kb_vec)
        if s > best_score:
            best_score = s
            best_idx = i
    if best_score >= 0.2 and best_idx is not None:
        return KB[best_idx][1], best_score
    else:
        return None, best_score


def main_loop():
    print("NLP Chatbot (type 'exit' or 'quit' to stop)")
    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user.strip():
            continue
        if user.lower().strip() in ('exit', 'quit'):
            print("Chatbot: Goodbye — have a nice day!")
            break

        intent = detect_intent(user)
        if intent == 'greeting':
            print("Chatbot: Hello! How can I help you today?")
            continue
        elif intent == 'goodbye':
            print("Chatbot: Goodbye!")
            break
        elif intent == 'thanks':
            print("Chatbot: You're welcome!")
            continue

        answer, score = find_best_answer(user)
        if answer:
            print(f"Chatbot: {answer} (confidence={score:.2f})")
        else:
            print("Chatbot: I'm not sure about that. You can extend the KB in kb.py.")


if __name__ == '__main__':
    main_loop()
