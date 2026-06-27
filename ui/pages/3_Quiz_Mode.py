import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.adaptive_engine.session_manager import AdaptiveSession
from modules.content_generator.flashcard_generator import generate_flashcards
from modules.content_generator.summary_generator import generate_summary

st.set_page_config(page_title="Quiz Mode", page_icon="📊", layout="wide")
st.title("📊 Adaptive Quiz Mode")
st.markdown("Questions automatically adjust to your level in real time.")

# Sidebar settings
with st.sidebar:
    st.subheader("⚙️ Quiz Settings")
    topic_name   = st.text_input("Topic name:", value="Physics")
    num_questions = st.slider("Number of questions:", 3, 15, 5)
    study_text   = st.text_area("Paste study material:", height=200,
                                placeholder="Paste the topic content here...")

    mode = st.radio("Mode", ["Adaptive Quiz", "Flashcards", "Summary"])

    start = st.button("Start", disabled=not study_text)

if not study_text:
    st.info("👈 Add your study material and click Start.")
    st.stop()

if start or st.session_state.get("quiz_active"):
    if mode == "Summary":
        with st.spinner("Generating summary..."):
            result = generate_summary(study_text, topic_name, style="bullet_points")
        st.subheader(f"📝 Summary: {topic_name}")
        st.write(result["summary"] if isinstance(result["summary"], str) else "\n".join(result["summary"]))
        st.divider()
        st.subheader("Key Concepts")
        for concept in result.get("key_concepts", []):
            st.markdown(f"- {concept}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Difficulty", result.get("difficulty_level", "unknown").title())
        with col2:
            st.metric("Study Time", result.get("estimated_study_time", "unknown"))

    elif mode == "Flashcards":
        with st.spinner("Generating flashcards..."):
            cards = generate_flashcards(study_text, topic_name, num_cards=8)

        st.subheader(f"🃏 Flashcards: {topic_name}")
        for i, card in enumerate(cards):
            with st.expander(f"Card {i+1} [{card.get('category', '')}]: {card['front']}"):
                st.success(card["back"])

    else:
        # Adaptive quiz
        if "session" not in st.session_state or start:
            st.session_state.session      = AdaptiveSession(study_text, topic_name, max_questions=num_questions)
            st.session_state.session.start()
            st.session_state.quiz_active  = True
            st.session_state.current_q    = None
            st.session_state.last_result  = None
            st.session_state.answered     = False

        session = st.session_state.session

        if session.is_done():
            summary = session.get_summary()
            st.balloons()
            st.success("Quiz Complete!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", summary["total_questions"])
            with col2:
                st.metric("Correct Answers", summary["correct_answers"])
            with col3:
                st.metric("Accuracy", f"{summary['overall_accuracy']:.0%}")
            st.info(f"Final difficulty reached: **{summary['final_difficulty']}**")
            if st.button("Start New Quiz"):
                for key in ["session", "quiz_active", "current_q", "last_result", "answered"]:
                    st.session_state.pop(key, None)
                st.rerun()
        else:
            if st.session_state.current_q is None or st.session_state.answered:
                st.session_state.current_q = session.next_question()
                st.session_state.answered  = False

            q = st.session_state.current_q
            if q:
                progress = session.questions_done / num_questions
                st.progress(progress, text=f"Question {session.questions_done + 1} of {num_questions}")

                diff_color = {
                    "very_easy": "🟢", "easy": "🔵",
                    "medium": "🟡", "hard": "🟠", "very_hard": "🔴"
                }.get(q.get("difficulty_level", "medium"), "⚪")

                st.subheader(f"{diff_color} {q['question']}")

                answer = st.radio("Choose your answer:", list(q["options"].keys()),
                                  format_func=lambda x: f"{x}) {q['options'][x]}")

                if st.button("Submit Answer"):
                    result = session.submit_answer(answer)
                    st.session_state.last_result = result
                    st.session_state.answered    = True

                    if result["is_correct"]:
                        st.success(f"✅ Correct! {result['explanation']}")
                    else:
                        st.error(f"❌ Wrong. Correct answer: {result['correct_answer']}")
                        st.info(result["explanation"])

                    st.info(f"💡 {result['feedback']} | New difficulty: **{result['new_difficulty']}**")

                    if st.button("Next Question"):
                        st.rerun()