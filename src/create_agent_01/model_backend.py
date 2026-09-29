import os
from typing import Optional

from create_agent_01.config import Settings

def get_chat_model(provider: str,
                   model_name: str,
                   **kwargs):
    
    provider = provider.lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
     
        model = ChatOllama(
               model = model_name,
               base_url = Settings.OLLAMA_BASE_URL,
               **kwargs
            )

        return model

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = Settings.GROQ_API_KEY
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env.")

        model = ChatGroq(model, api_key, **kwargs)
        return model

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        pipeline = HuggingFacePipeline.from_model_id(
               model = model_name,
               task="text-generation",
               pipeline_kwargs={"max_new_tokens": 512,"do_sample": False},
             )

        return ChatHuggingFace(llm=pipeline)

    raise ValueError(
        f"Unknown backend '{provider}'. Choose 'ollama', 'groq', or 'huggingface_local'."
    )

def get_embeddings(provder: str,
                   model: Optional[str]=None):
    if provder.lower() == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model or "nomic-embed-text",
            base_url=Settings.OLLAMA_BASE_URL
        )

    if provder.lower() == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name = model or "sentence-transformers/all-MiniLM-L6-v2"
        )

    raise ValueError(
            "Groq has no embeddings endpoint. Choose 'ollama' or 'huggingface_local' "
            "for embeddings."
        )

def enable_langsmith_tracing(project_name: str) -> None:
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = project_name
    else:
        print(
            f"[warn] LANGSMITH_API_KEY not set - '{project_name}' will run without tracing."
        )
