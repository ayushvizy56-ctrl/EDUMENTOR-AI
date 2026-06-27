import json
from groq import Groq
from config.settings import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)


def generate_flashcards(text: str, topic_name: str, num_cards: int = 8) -> list[dict]:
    """
    Generates flashcards from study material.
    Each flashcard has a front (question/term) and back (answer/definition).

    Returns:
        list of {"front": "...", "back": "...", "category": "..."}
    """
    prompt = f"""You are an expert teacher creating flashcards for "{topic_name}".

Generate exactly {num_cards} flashcards from the study material below.
Mix different types: definitions, formulas, concepts, and examples.

STUDY MATERIAL:
{text[:3000]}

Return ONLY this JSON format, no extra text:
[
  {{
    "front": "What is...? / Define... / What does... mean?",
    "back": "Clear, concise answer in 1-2 sentences",
    "category": "definition / formula / concept / example"
  }}
]"""

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        cards = json.loads(raw)
        print(f"Generated {len(cards)} flashcards for '{topic_name}'")
        return cards
    except json.JSONDecodeError:
        print("Warning: Could not parse flashcard JSON.")
        return []


if __name__ == "__main__":
    text = """
    Newton's Laws of Motion:
    First Law (Inertia): An object at rest stays at rest, an object in motion stays in motion
    unless acted upon by an external force.
    Second Law: F = ma. Force equals mass times acceleration.
    Third Law: For every action there is an equal and opposite reaction.

    Gravity: F = GMm/r² where G is gravitational constant, M and m are masses, r is distance.
    Weight = mass × gravitational acceleration (W = mg, where g = 9.8 m/s² on Earth)
    """

    cards = generate_flashcards(text, "Newton's Laws", num_cards=5)
    print()
    for i, card in enumerate(cards, 1):
        print(f"Card {i} [{card['category']}]")
        print(f"  Front: {card['front']}")
        print(f"  Back:  {card['back']}")
        print()