import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONTAMINATION_RATE = 0.1
HIGH_RISK_THRESHOLD = 71
MEDIUM_RISK_THRESHOLD = 41
HIGH_VALUE_ITEM_THRESHOLD = 500
MODEL_DIR = "models"
MODEL_PATH = "models/isolation_forest.pkl"
USER_FEATURES_PATH = "models/user_features.pkl"
DATA_PATH = "data/processed/returns_fraud_dataset.csv"
API_BASE_URL = "http://localhost:8001/api/v1"
API_TIMEOUT = 1
MIN_RETURN_AGE_DAYS = 30
MAX_RISK_SCORE = 100

# RAG Chatbot Settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(MODEL_DIR, "chroma_db"))
