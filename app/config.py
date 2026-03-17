import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_LARGE_MODEL = "mistral-large-latest"
MISTRAL_SMALL_MODEL = "mistral-small-latest"
MISTRAL_EMBED_MODEL = "mistral-embed"
CONFIDENCE_THRESHOLD = 0.85
SIMILARITY_THRESHOLD = 0.3
MAX_CONVERSATION_TURNS = 10
