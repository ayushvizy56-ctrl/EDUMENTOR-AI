import streamlit as st

st.set_page_config(
    page_title="EduMentor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — clean, professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #4361ee;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4361ee;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.2rem;
    }
    .stButton > button {
        background: #4361ee;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: #3651d4;
    }
    div[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎓 EduMentor AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your personalized AI-powered learning companion</p>', unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:1.8rem">🧠</div>
        <div class="metric-value">ML</div>
        <div class="metric-label">Knowledge Profiling — tracks what you know and where to improve</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:1.8rem">💬</div>
        <div class="metric-value">RAG</div>
        <div class="metric-label">AI Chatbot — answers questions from your own study material</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:1.8rem">🤖</div>
        <div class="metric-value">Agent</div>
        <div class="metric-label">Study Planner — autonomous AI that builds your study schedule</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

col4, col5 = st.columns(2)
with col4:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:1.8rem">📊</div>
        <div class="metric-value">Adaptive</div>
        <div class="metric-label">Quiz Engine — adjusts difficulty based on your performance in real time</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:1.8rem">✍️</div>
        <div class="metric-value">GenAI</div>
        <div class="metric-label">Content Generator — creates quizzes, summaries, flashcards from your notes</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("### 👈 Use the sidebar to navigate between modules")
st.info("Start with **Knowledge Profile** to set up your student profile, then explore other features.")