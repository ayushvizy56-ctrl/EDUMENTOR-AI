import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

ROOT_DIR  = Path(__file__).parent.parent
DATA_RAW  = ROOT_DIR / "data" / "raw"
DATA_PROC = ROOT_DIR / "data" / "processed"
EMBED_DIR = ROOT_DIR / "data" / "embeddings"
MODEL_DIR = ROOT_DIR / "modules" / "knowledge_profiling" / "models"

TOPICS = [
    "algebra", "geometry", "trigonometry",
    "physics_mechanics", "physics_optics",
    "chemistry_organic", "chemistry_inorganic",
    "biology_cell", "biology_genetics",
    "history", "geography", "english_grammar",
]

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_STUDENTS   = 2000
QUIZZES_EACH = 10