import sys
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph

from config import Settings
from rag import build_retriever_tool
from model_backend import default_llm, enable_langsmith_tracing

class ChainState(TypedDict, total=False):
    question: str
    context: str
    draft: str
    answer: str

llm = default_llm()
kb_search = build_retriever_tool()

def retrieve(state: ChainState) -> ChainState:
    resp = kb_search.invoke({"query": state["question"]})
    return {"context": resp}

def draft(state: ChainState) -> ChainState:
    msg = llm.invoke([
        SystemMessage(content=(
            "Answer only from the context. FInish with a line 'Source: <file name?' "
            "using the [source: ...] tags in the context. If the context does not "
            "answer the question, say so."
        )),
        HumanMessage(content=f"Context=\n{state["context"]}\n\nQuestion:\n{state["question"]}"),
    ])

    return {"draft": msg.content}

def has_source(state: ChainState) -> str:
    print(state["draft"])
    return "polish" if "source" in state["draft"].lower() else "add_source"

def add_source(state: ChainState) -> ChainState:
    msg = llm.invoke([
        SystemMessage(content="Append a 'Source: <file name>' line to the draft using the context tags. Change nothing else."),
        HumanMessage(content=f"Context:\n{state['context']}\n\nDraft:\n{state['draft']}"),
    ])
    return {"draft": msg.content}

def polish(state: ChainState) -> ChainState:
    msg = llm.invoke([
        SystemMessage(content="Rewrite as at most 3 short bullet points for an employee. Keep every number and the source line exactly."),
        HumanMessage(content=state["draft"]),
    ])
    return {"answer": msg.content}

def build_graph():
    g=StateGraph(ChainState)
    
    g.add_node("retrieve", retrieve)
    g.add_node("draft", draft)
    g.add_node("add_source", add_source)
    g.add_node("polish", polish)

    g.add_edge(START, "retrieve")
    g.add_edge( "retrieve", "draft")
    g.add_conditional_edges("draft", has_source, ["polish", "add_source"])
    g.add_edge("add_source", "polish")
    g.add_edge("polish", END)
  
    return g.compile()

def main():
    graph = build_graph()
    if "--mermaid" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        return
    enable_langsmith_tracing(Settings.LANGCHAIN_PROJECT)
    result = graph.invoke({"question": "What are the rules for claiming meal expenses?"})
    print(result["answer"])

if __name__ == "__main__":
    main()