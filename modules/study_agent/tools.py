"""
Tools that the Study Planner Agent can call.
Each tool does one specific job — the agent decides which to call and when.
This is the core of agentic AI: giving an LLM the ability to take actions.
"""

import json
from groq import Groq
from config.settings import GROQ_API_KEY, TOPICS
from modules.content_generator.summary_generator import generate_summary
from modules.content_generator.quiz_generator import generate_quiz
from modules.content_generator.flashcard_generator import generate_flashcards

_client = Groq(api_key=GROQ_API_KEY)


def tool_assess_student(student_profile: dict) -> dict:
    """
    Analyzes a student's knowledge profile and identifies
    priority topics that need the most attention.
    """
    weak_topics  = student_profile.get("weak_topics", [])
    topic_scores = student_profile.get("topic_scores", {})
    performance  = student_profile.get("performance", "average")
    trend        = student_profile.get("score_trend", 0)

    # Sort topics by score (lowest first = most urgent)
    sorted_topics = sorted(
        topic_scores.items(),
        key=lambda x: x[1]
    )

    priority_topics = [t for t, s in sorted_topics[:4]]
    strong_topics   = [t for t, s in sorted_topics if s > 0.7]

    return {
        "tool":            "assess_student",
        "performance":     performance,
        "improving":       trend > 0,
        "priority_topics": priority_topics,
        "strong_topics":   strong_topics,
        "weak_count":      len(weak_topics),
        "insight":         f"Student is {performance} and {'improving' if trend > 0 else 'needs motivation'}. "
                           f"Focus areas: {', '.join(priority_topics[:3])}.",
    }


def tool_create_study_plan(
    student_profile: dict,
    available_hours_per_day: float,
    exam_days_away: int,
    priority_topics: list
) -> dict:
    """
    Creates a day-by-day study plan tailored to the student.
    Uses Groq LLM to generate an intelligent, personalized schedule.
    """
    topic_scores  = student_profile.get("topic_scores", {})
    performance   = student_profile.get("performance", "average")

    scores_text = "\n".join([
        f"  - {topic}: {score:.0%}"
        for topic, score in topic_scores.items()
        if topic in priority_topics
    ])

    prompt = f"""You are an expert academic planner creating a personalized study plan.

STUDENT PROFILE:
- Performance level: {performance}
- Available study time: {available_hours_per_day} hours/day
- Days until exam: {exam_days_away}
- Topics needing focus:
{scores_text}

Create a realistic {min(exam_days_away, 7)}-day study plan.

Rules:
- Weaker topics get more time
- Include breaks and revision days
- Be specific about time allocation per topic per day
- Day 1 should start with the weakest topic

Return ONLY this JSON:
{{
  "plan_title": "...",
  "total_days": {min(exam_days_away, 7)},
  "daily_schedule": [
    {{
      "day": 1,
      "date_label": "Day 1",
      "focus_topic": "algebra",
      "sessions": [
        {{"time": "Morning", "topic": "algebra", "activity": "Review basics", "duration_min": 45}},
        {{"time": "Evening", "topic": "algebra", "activity": "Practice problems", "duration_min": 30}}
      ],
      "daily_goal": "Understand and practice basic algebraic equations"
    }}
  ],
  "tips": ["tip 1", "tip 2", "tip 3"]
}}"""

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.4,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        plan = json.loads(raw)
    except Exception:
        plan = {"plan_title": "Study Plan", "raw": raw}

    plan["tool"] = "create_study_plan"
    return plan


def tool_generate_resources(topic: str, topic_text: str) -> dict:
    """
    Generates a complete study resource pack for one topic:
    summary + flashcards + 3 practice questions.
    """
    print(f"  Generating resources for: {topic}")

    summary    = generate_summary(topic_text, topic, style="bullet_points")
    flashcards = generate_flashcards(topic_text, topic, num_cards=5)
    questions  = generate_quiz(topic_text, topic, num_questions=3, difficulty="medium")

    return {
        "tool":        "generate_resources",
        "topic":       topic,
        "summary":     summary,
        "flashcards":  flashcards,
        "questions":   questions,
    }


def tool_check_progress(session_results: list[dict]) -> dict:
    """
    Analyzes quiz session results and gives the agent
    insight into whether the student is improving.
    """
    if not session_results:
        return {"tool": "check_progress", "status": "no_data"}

    total    = len(session_results)
    correct  = sum(1 for r in session_results if r.get("is_correct"))
    accuracy = correct / total if total > 0 else 0

    # Check if accuracy is trending up in recent sessions
    if total >= 2:
        first_half  = sum(1 for r in session_results[:total//2] if r.get("is_correct")) / (total//2)
        second_half = sum(1 for r in session_results[total//2:] if r.get("is_correct")) / (total - total//2)
        improving   = second_half > first_half
    else:
        improving = False

    status = "excellent" if accuracy > 0.8 else "good" if accuracy > 0.6 else "needs_work"

    return {
        "tool":       "check_progress",
        "accuracy":   round(accuracy, 3),
        "status":     status,
        "improving":  improving,
        "recommendation": (
            "Move to the next topic" if status == "excellent" else
            "Review weak areas before moving on" if status == "good" else
            "Spend more time on fundamentals"
        ),
    }