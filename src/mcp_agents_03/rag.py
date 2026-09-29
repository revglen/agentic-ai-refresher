import os

from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_backend import get_embeddings

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

def _load_corpus_texts() -> list[tuple[str, str]]:
    """Returns a list (filename, text) for every .txt file in corpus/."""
 
    texts=[]
 
    for fname in sorted(os.listdir(CORPUS_DIR)):
        if fname.endswith(".txt"):
            path=os.path.join(CORPUS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                texts.append((fname, f.read()))

    return texts

def build_retriever_tool(embedding_provider: str="ollama", k: int = 3):
    """
    Builds the vector store from shared/corpus/ and returns a ready-to-use
    LangChain tool named "knowledge_base_search".
    """

    embeddings = get_embeddings(embedding_provider)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs=[]
     
    for fname, text in _load_corpus_texts():
       for chunk in splitter.split_text(text):
         docs.append({"text": chunk, "source": fname})

    if not docs:
        raise RuntimeError(f"No .txt files found in {CORPUS_DIR}")

    vector_store = InMemoryVectorStore.from_texts(
        texts=[d["text"] for d in docs],
        embedding=embeddings,
        metadatas=[{"source": d["source"]} for d in docs]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return create_retriever_tool(
       retriever,
       name="knowledge_base_search",
       description=(
            "Search the internal sample knowledge base (remote work policy, "
            "expense policy, onboarding guide) for a passage relevant to the "
            "query. Use this before answering any question about company "
            "policy."
        ),
    )