from groq import Groq
from config.settings import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)


def generate_summary(text: str, topic_name: str, style: str = "concise") -> dict:
    """
    Generates a structured summary of study material.

    Args:
        text:       the study material
        topic_name: name of the topic
        style:      concise / detailed / bullet_points

    Returns:
        dict with summary, key_points, and difficulty_level
    """
    style_instruction = {
        "concise":       "Write a clear 3-4 sentence summary capturing the core concept.",
        "detailed":      "Write a comprehensive paragraph summary covering all key aspects.",
        "bullet_points": "Write a bullet-point summary with 5-7 key points, each one sentence.",
    }.get(style, "Write a clear concise summary.")

    prompt = f"""You are an expert educational content creator.
Summarize the following study material on "{topic_name}".

{style_instruction}

Also identify:
- 3-5 key concepts students must remember
- The overall difficulty level (beginner/intermediate/advanced)

STUDY MATERIAL:
{text[:3000]}

Respond in this exact JSON format:
{{
  "summary": "...",
  "key_concepts": ["concept 1", "concept 2", "concept 3"],
  "difficulty_level": "intermediate",
  "estimated_study_time": "15 minutes"
}}

Return ONLY the JSON, no extra text."""

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = __import__("json").loads(raw)
        print(f"Summary generated for '{topic_name}'")
        return result
    except Exception:
        return {
            "summary":              raw,
            "key_concepts":         [],
            "difficulty_level":     "unknown",
            "estimated_study_time": "unknown",
        }


if __name__ == "__main__":
    text = """
    The mitochondria is often called the powerhouse of the cell. It is a double-membraned
    organelle found in the cytoplasm of eukaryotic cells. The main function of mitochondria
    is to produce ATP (adenosine triphosphate) through a process called cellular respiration.

    Cellular respiration occurs in three main stages:
    1. Glycolysis — occurs in the cytoplasm, breaks glucose into pyruvate
    2. Krebs Cycle — occurs in the mitochondrial matrix, produces CO2 and electron carriers
    3. Electron Transport Chain — occurs in the inner mitochondrial membrane, produces most ATP

    Mitochondria have their own DNA and can replicate independently, which supports
    the endosymbiotic theory — the idea that mitochondria were once free-living bacteria.
    """

    result = generate_summary(text, "Mitochondria", style="bullet_points")
    print("\nSummary:", result["summary"])
    print("\nKey Concepts:", result["key_concepts"])
    print("Difficulty:", result["difficulty_level"])
    print("Study Time:", result["estimated_study_time"])