import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.study_agent.planner import create_personalized_plan

st.set_page_config(page_title="Study Planner", page_icon="🤖", layout="wide")
st.title("🤖 AI Study Planner")
st.markdown("The autonomous AI agent analyzes your profile and builds a personalized study plan.")

# Check if profile exists from page 1
profile = st.session_state.get("student_profile")

if not profile:
    st.warning("No student profile found. Please go to **Knowledge Profile** first, or use the demo below.")
    if st.button("Use Demo Profile"):
        profile = {
            "performance":   "average",
            "confidence":    0.78,
            "avg_score":     0.55,
            "score_trend":   0.01,
            "weak_topics":   ["algebra", "chemistry_organic", "physics_mechanics"],
            "topic_scores":  {
                "algebra": 0.35, "geometry": 0.60, "trigonometry": 0.52,
                "physics_mechanics": 0.40, "physics_optics": 0.68,
                "chemistry_organic": 0.30, "chemistry_inorganic": 0.62,
                "biology_cell": 0.70, "biology_genetics": 0.65,
                "history": 0.78, "geography": 0.72, "english_grammar": 0.82,
            },
            "recommendation": "Focus on algebra and chemistry.",
        }
        st.session_state["student_profile"] = profile
        st.rerun()
    st.stop()

# Settings
st.subheader("⚙️ Plan Settings")
col1, col2 = st.columns(2)
with col1:
    hours_per_day = st.slider("Study hours per day:", 0.5, 6.0, 2.0, step=0.5)
with col2:
    exam_days = st.slider("Days until exam:", 3, 30, 7)

# Topic texts for resource generation
st.subheader("📚 Add Study Material (optional)")
st.caption("Add text for your weakest topics so the agent can generate resources.")

topic_texts = {}
for topic in profile.get("weak_topics", [])[:2]:
    text = st.text_area(f"Notes for {topic.replace('_', ' ').title()}:",
                        placeholder=f"Paste your {topic} notes here...",
                        height=100, key=f"topic_{topic}")
    if text:
        topic_texts[topic] = text

st.divider()

if st.button("🚀 Generate My Study Plan", type="primary"):
    with st.spinner("AI Agent is thinking and planning... (this takes 30-60 seconds)"):
        try:
            result = create_personalized_plan(
                student_profile = profile,
                topic_texts     = topic_texts,
                available_hours = hours_per_day,
                exam_days       = exam_days,
            )
            st.session_state["plan_result"] = result
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

if "plan_result" in st.session_state:
    result = st.session_state["plan_result"]
    tool_results = result.get("tool_results", {})

    st.success(f"Plan generated in {result['steps_taken']} steps!")

    # Study plan
    if "create_study_plan" in tool_results:
        plan = tool_results["create_study_plan"]
        st.subheader(f"📅 {plan.get('plan_title', 'Your Study Plan')}")

        for day in plan.get("daily_schedule", []):
            with st.expander(f"📆 {day['date_label']} — {day['focus_topic'].replace('_', ' ').title()}"):
                for session in day.get("sessions", []):
                    st.markdown(f"**{session['time']}:** {session['activity']} _{session['duration_min']} min_")
                st.info(f"🎯 Goal: {day['daily_goal']}")

        if plan.get("tips"):
            st.subheader("💡 Tips")
            for tip in plan["tips"]:
                st.markdown(f"- {tip}")

    # Generated resources
    if "generate_resources" in tool_results:
        resources = tool_results["generate_resources"]
        topic     = resources.get("topic", "")
        st.divider()
        st.subheader(f"📚 Resources for {topic.replace('_', ' ').title()}")

        tab1, tab2, tab3 = st.tabs(["Summary", "Flashcards", "Practice Questions"])

        with tab1:
            summary = resources.get("summary", {})
            if isinstance(summary.get("summary"), list):
                for point in summary["summary"]:
                    st.markdown(f"- {point}")
            else:
                st.write(summary.get("summary", ""))

        with tab2:
            for i, card in enumerate(resources.get("flashcards", []), 1):
                with st.expander(f"Card {i}: {card['front']}"):
                    st.success(card["back"])

        with tab3:
            for i, q in enumerate(resources.get("questions", []), 1):
                st.markdown(f"**Q{i}: {q['question']}**")
                for k, v in q["options"].items():
                    st.markdown(f"  {k}) {v}")
                with st.expander("Show Answer"):
                    st.success(f"Answer: {q['correct_answer']}")
                    st.info(q["explanation"])
                st.divider()

    # Agent reasoning
    with st.expander("🧠 Agent Reasoning Trace"):
        for step in result.get("reasoning", []):
            st.markdown(f"**Step {step['step']} [{step['action']}]:** {step['thought']}")