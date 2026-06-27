import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.knowledge_profiling.predictor import predict

st.set_page_config(page_title="Knowledge Profile", page_icon="🧠", layout="wide")
st.title("🧠 Knowledge Profiling")
st.markdown("Enter your quiz history and get a full analysis of your knowledge gaps.")

st.divider()

# Quick demo mode
st.subheader("Try with Demo Data")
if st.button("Load Demo Student"):
    st.session_state["demo_loaded"] = True

if st.session_state.get("demo_loaded"):
    from modules.knowledge_profiling.data_generator import generate_dataset
    import numpy as np

    # Generate a single student's quiz history
    np.random.seed(99)
    demo_history = []
    topics = ["algebra", "geometry", "physics_mechanics", "chemistry_organic", "history", "english_grammar"]
    for i, topic in enumerate(topics * 2):
        demo_history.append({
            "student_id":     0,
            "quiz_number":    i + 1,
            "topic":          topic,
            "difficulty":     round(np.random.uniform(0.3, 0.8), 2),
            "time_spent_sec": round(np.random.uniform(20, 60), 1),
            "hints_used":     np.random.randint(0, 3),
            "score":          round(np.random.choice([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]), 1),
        })

    with st.spinner("Analyzing your knowledge profile..."):
        try:
            profile = predict(demo_history)
            st.session_state["student_profile"] = profile
            st.success("Profile generated!")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    profile = st.session_state.get("student_profile", {})
    if not profile:
        st.stop()

    # Show performance badge
    perf_color = {"weak": "🔴", "average": "🟡", "strong": "🟢"}.get(profile["performance"], "⚪")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Performance Level", f"{perf_color} {profile['performance'].title()}")
    with col2:
        st.metric("Average Score", f"{profile['avg_score']:.0%}")
    with col3:
        trend_icon = "📈" if profile["score_trend"] > 0 else "📉"
        st.metric("Trend", f"{trend_icon} {'Improving' if profile['score_trend'] > 0 else 'Declining'}")

    st.divider()

    # Topic scores radar chart
    topic_scores = profile["topic_scores"]
    topics_list  = list(topic_scores.keys())
    scores_list  = list(topic_scores.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=scores_list,
        theta=[t.replace("_", " ").title() for t in topics_list],
        fill="toself",
        fillcolor="rgba(67, 97, 238, 0.2)",
        line=dict(color="#4361ee", width=2),
        name="Your Scores",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Topic Knowledge Radar",
        showlegend=False,
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weak topics
    if profile["weak_topics"]:
        st.subheader("⚠️ Topics Needing Attention")
        for topic in profile["weak_topics"]:
            score = topic_scores.get(topic, 0)
            st.progress(score, text=f"{topic.replace('_', ' ').title()} — {score:.0%}")

    st.divider()
    st.subheader("💡 Recommendation")
    st.info(profile["recommendation"])
    st.success("Profile saved! Go to **Study Planner** to create your personalized plan.")