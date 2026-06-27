"""
High-level interface for the Study Planner Agent.
This is what the Streamlit UI and FastAPI will call.
"""

from modules.study_agent.agent import StudyPlannerAgent


def create_personalized_plan(
    student_profile: dict,
    topic_texts: dict,
    available_hours: float = 2.0,
    exam_days: int = 7,
) -> dict:
    """
    Main entry point — creates a complete personalized study plan.

    Args:
        student_profile: output from modules.knowledge_profiling.predictor.predict()
        topic_texts:     dict of {topic_name: study_text} for resource generation
        available_hours: hours per day the student can study
        exam_days:       days until the exam

    Returns:
        Complete study plan with schedule, resources, and agent reasoning
    """
    agent = StudyPlannerAgent(max_steps=6)

    goal = (
        f"Create a complete {exam_days}-day personalized study plan for this student. "
        f"They have {available_hours} hours/day to study. "
        f"Assess their profile, build a schedule, and generate resources for their weakest topic."
    )

    context = {
        "student_profile": student_profile,
        "topic_texts":     topic_texts,
        "available_hours": available_hours,
        "exam_days":       exam_days,
    }

    result = agent.run(goal, context)
    return result


if __name__ == "__main__":
    # Test the full agent with a mock student profile
    mock_profile = {
        "performance":   "average",
        "confidence":    0.82,
        "avg_score":     0.58,
        "score_trend":   0.012,
        "weak_topics":   ["algebra", "chemistry_organic", "physics_mechanics"],
        "topic_scores":  {
            "algebra":             0.35,
            "geometry":            0.62,
            "trigonometry":        0.55,
            "physics_mechanics":   0.40,
            "physics_optics":      0.70,
            "chemistry_organic":   0.30,
            "chemistry_inorganic": 0.65,
            "biology_cell":        0.72,
            "biology_genetics":    0.68,
            "history":             0.80,
            "geography":           0.75,
            "english_grammar":     0.85,
        },
        "recommendation": "Focus on algebra and chemistry.",
    }

    topic_texts = {
        "algebra": """
            Algebra is the branch of mathematics dealing with symbols and rules for
            manipulating those symbols. Key concepts include variables, expressions,
            equations, and functions. Linear equations have the form ax + b = 0.
            Quadratic equations have the form ax² + bx + c = 0 and can be solved
            using the quadratic formula: x = (-b ± √(b²-4ac)) / 2a.
            """,
        "chemistry_organic": """
            Organic chemistry is the study of carbon-containing compounds.
            Hydrocarbons contain only carbon and hydrogen. Alkanes are saturated
            hydrocarbons with single bonds (CnH2n+2). Alkenes contain double bonds
            (CnH2n). Alkynes contain triple bonds (CnH2n-2). Functional groups
            like hydroxyl (-OH), carbonyl (C=O), and carboxyl (-COOH) determine
            the properties and reactions of organic molecules.
            """,
    }

    print("Running Study Planner Agent...\n")
    result = create_personalized_plan(
        student_profile = mock_profile,
        topic_texts     = topic_texts,
        available_hours = 2.0,
        exam_days       = 5,
    )

    print(f"\nAgent completed in {result['steps_taken']} steps")
    print("\nReasoning trace:")
    for step in result["reasoning"]:
        print(f"  Step {step['step']}: [{step['action']}] {step['thought'][:80]}")

    # Show the study plan if it was generated
    if "create_study_plan" in result["tool_results"]:
        plan = result["tool_results"]["create_study_plan"]
        print(f"\n{'='*50}")
        print(f"STUDY PLAN: {plan.get('plan_title', 'Your Plan')}")
        print(f"{'='*50}")
        for day in plan.get("daily_schedule", [])[:3]:
            print(f"\n{day['date_label']} — Focus: {day['focus_topic']}")
            for session in day.get("sessions", []):
                print(f"  {session['time']}: {session['activity']} ({session['duration_min']} min)")
            print(f"  Goal: {day['daily_goal']}")