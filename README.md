# EduMentor AI 🎓
### Intelligent Personalized Learning Platform

An end-to-end AI-powered education system combining Machine Learning, 
Deep Learning, NLP, Generative AI, and Agentic AI into one cohesive product.

## Architecture
- **Module 1 — Knowledge Profiling** (ML): XGBoost classifier that tracks student 
  performance across 12 subjects and predicts weak areas with 99% accuracy
- **Module 2 — Emotion Detection** (Deep Learning): Real-time engagement tracking 
  using DeepFace — detects if a student is focused, distracted, or frustrated
- **Module 3 — RAG Chatbot** (NLP + GenAI): Answers student questions from their 
  own uploaded study material using FAISS vector search + Groq LLaMA 3.3
- **Module 4 — Content Generator** (GenAI): Auto-generates quizzes, summaries, 
  and flashcards from any study material using structured LLM prompting
- **Module 5 — Adaptive Engine** (ML): Adjusts question difficulty in real time 
  based on rolling accuracy — similar to how Duolingo and Khan Academy work
- **Module 6 — Study Planner Agent** (Agentic AI): ReAct-pattern autonomous agent 
  that assesses student profile, builds study schedules, and generates resources

## Tech Stack
| Layer | Technologies |
|---|---|
| ML / DL | Scikit-learn, XGBoost, TensorFlow, DeepFace |
| NLP / GenAI | LangChain, FAISS, Sentence Transformers, Groq API |
| Agentic AI | Custom ReAct Agent, LangGraph |
| Backend | FastAPI, Python |
| Frontend | Streamlit, Plotly |
| Data | Pandas, NumPy |

## How to Run
```bash
# 1. Clone and setup
git clone https://github.com/yourusername/EduMentor-AI
cd EduMentor-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your API key
echo "GROQ_API_KEY=your_key" > config/.env

# 3. Train the knowledge model
python -m modules.knowledge_profiling.data_generator
python -m modules.knowledge_profiling.feature_engineering  
python -m modules.knowledge_profiling.train_model

# 4. Launch the app
streamlit run ui/app.py
```

## Project Structure
```
EduMentor-AI/
├── modules/
│   ├── knowledge_profiling/   # ML — student knowledge tracking
│   ├── emotion_detection/     # DL — engagement monitoring  
│   ├── rag_chatbot/           # NLP — document Q&A
│   ├── content_generator/     # GenAI — quiz/summary/flashcard
│   ├── adaptive_engine/       # ML — difficulty adjustment
│   └── study_agent/           # Agentic AI — autonomous planner
├── ui/                        # Streamlit dashboard
├── api/                       # FastAPI backend
├── data/                      # Datasets and embeddings
└── config/                    # Settings and API keys
```