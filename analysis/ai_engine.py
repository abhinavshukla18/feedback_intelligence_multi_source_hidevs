import os
import time
import groq
from groq import Groq
from dotenv import load_dotenv

from analysis.embeddings import load_model, search_reviews

load_dotenv()  # reads GROQ_API_KEY from the .env file

MODEL_NAME = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are a product analyst helping a product manager understand customer "
    "feedback about the Spotify app. You will be given numbered customer "
    "reviews and a question. Rules: use ONLY the reviews provided; mention "
    "the review numbers you relied on, like [2] or [5]; if the reviews do not "
    "contain enough information to answer, say so plainly; keep the answer "
    "short and specific."
)


def build_prompt(question, hits):
    """Put the retrieved reviews and the question into one message."""
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] ({hit['source']}, {hit['rating']:.0f} stars) {hit['text']}")
    reviews_block = "\n".join(lines)
    return f"Customer reviews:\n{reviews_block}\n\nQuestion: {question}"


def call_llm(messages, retries=3):
    """Send messages to Groq. Retries on rate limits, falls back to a smaller model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY was not found. Check your .env file."

    client = Groq(api_key=api_key)

    for model in [MODEL_NAME, FALLBACK_MODEL]:
        for attempt in range(1, retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,   # low = focused, less creative
                    max_tokens=1500,
                )
                return response.choices[0].message.content or "The model returned an empty answer."
            except groq.RateLimitError:
                print(f"Rate limit hit ({model}), attempt {attempt}. Waiting...")
                time.sleep(5 * attempt)
            except groq.APIConnectionError:
                print(f"Network problem ({model}), attempt {attempt}.")
                time.sleep(2 * attempt)
            except groq.APIStatusError as error:
                print(f"Groq error with {model}: {error}")
                break  # no point retrying this model, try the next one

    return "Sorry, the AI engine could not answer right now. Please try again."


def ask_reviews(question, embed_model=None, n_results=8, source=None):
    """Retrieve relevant reviews, then ask the LLM to answer from them."""
    hits = search_reviews(question, model=embed_model,
                          n_results=n_results, source=source)
    if not hits:
        return {"answer": "No relevant reviews were found.", "reviews": []}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},
    ]
    return {"answer": call_llm(messages), "reviews": hits}


if __name__ == "__main__":
    embed_model = load_model()
    questions = [
        "What are users complaining about after the latest update?",
        "What do people say about login problems?",
    ]
    for q in questions:
        result = ask_reviews(q, embed_model=embed_model)
        print(f"\nQUESTION: {q}")
        print(f"ANSWER: {result['answer']}")
        print(f"(based on {len(result['reviews'])} retrieved reviews)")