import json
from groq import Groq
from config.settings import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)


def generate_quiz(topic_text: str, topic_name: str, num_questions: int = 5, difficulty: str = "medium") -> list[dict]:
    """
    Generates MCQ quiz questions from study material.

    Args:
        topic_text:     the study material text
        topic_name:     name of the topic (e.g. "Newton's Laws")
        num_questions:  how many questions to generate
        difficulty:     easy / medium / hard

    Returns:
        list of question dicts with question, options, answer, explanation
    """
    prompt = f"""You are an expert teacher creating a quiz on "{topic_name}".

Generate exactly {num_questions} multiple choice questions at {difficulty} difficulty level.

STUDY MATERIAL:
{topic_text[:3000]}

STRICT JSON FORMAT — return ONLY this, no extra text:
[
  {{
    "question": "What is...?",
    "options": {{
      "A": "First option",
      "B": "Second option", 
      "C": "Third option",
      "D": "Fourth option"
    }},
    "correct_answer": "A",
    "explanation": "Because..."
  }}
]

Rules:
- Questions must be based ONLY on the study material
- Each question must have exactly 4 options (A, B, C, D)
- Only one correct answer per question
- Explanation must reference the material
- {difficulty} difficulty means {"basic recall" if difficulty == "easy" else "understanding and application" if difficulty == "medium" else "analysis and evaluation"}"""

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up markdown code blocks if present
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        questions = json.loads(raw)
        print(f"Generated {len(questions)} questions on '{topic_name}'")
        return questions
    except json.JSONDecodeError:
        print("Warning: Could not parse quiz JSON. Returning empty list.")
        return []


def evaluate_answer(question: dict, student_answer: str) -> dict:
    """
    Checks if a student's answer is correct and returns feedback.
    """
    is_correct = student_answer.upper() == question["correct_answer"].upper()
    return {
        "is_correct":      is_correct,
        "correct_answer":  question["correct_answer"],
        "explanation":     question["explanation"],
        "feedback":        "Correct! " + question["explanation"] if is_correct
                           else f"Not quite. The correct answer is {question['correct_answer']}. {question['explanation']}"
    }


if __name__ == "__main__":
    sample_text = """
    Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide
    to produce oxygen and energy in the form of glucose. This process occurs in the chloroplasts,
    specifically using chlorophyll — the green pigment that absorbs light energy.

    The overall equation for photosynthesis is:
    6CO2 + 6H2O + light energy → C6H12O6 + 6O2

    There are two main stages:
    1. Light-dependent reactions: occur in the thylakoid membranes, convert light to chemical energy (ATP and NADPH)
    2. Light-independent reactions (Calvin cycle): occur in the stroma, use ATP and NADPH to fix CO2 into glucose
    """

    questions = generate_quiz(sample_text, "Photosynthesis", num_questions=3, difficulty="medium")

    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q['question']}")
        for key, val in q["options"].items():
            print(f"  {key}) {val}")
        print(f"Answer: {q['correct_answer']}")
        print(f"Explanation: {q['explanation']}")