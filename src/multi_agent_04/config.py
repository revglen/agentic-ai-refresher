import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY=os.getenv("GROQ_API_KEY", "")
    OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "")
    LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT", "agentic-refresher-1")
    LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT", "https://eu.api.smith.langchain.com")
    
    HF_TOKEN=os.getenv("HF_TOKEN", "")
    PROVIDER="ollama"
    MODEL_NAME="qwen2.5:7b"

    EMBEDDING_PROVIDER=os.getenv("EMBEDDING_PROVIDER", "ollama")