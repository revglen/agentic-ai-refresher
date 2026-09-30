import operator
import sys
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from config import Settings
from rag import build_retriever_tool
from model_backend import default_llm, enable_langsmith_tracing

class Section(BaseModel):
    title: str
    search_query: str = Field(description="What to look up in the policy knowledge base for this section.")

class Plan(BaseModel):
    sections: list[Section] = Field(description="Between 2 and 5 sections.")

class BriefState(TypedDict, total=False):
    task: str
    sections: list[Section]
    completed: Annotated[list[str], operator.add]
    brief: str

class WorkerState(TypedDict):
    section: Section
    completed: Annotated[list[str], operator.add]

llm=default_llm()
kb_search=build_retriever_tool()

def orchestrator(state: BriefState) -> BriefState:
    plan = llm.with_structured_output(Plan).invoke ([
         SystemMessage(content="Plan the sections of the requested document.Sources available: remote work policy, expense policy, onboarding guide."),
        HumanMessage(content=state["task"]),
    ])

    for section in plan.sections:
        print(section)
  
    return {"sections": plan.sections}

def assign_workers(state: BriefState) -> list[Send]:
    return [Send("worker", {"section": s}) for s in state["sections"]]

def worker(state: WorkerState) -> dict:
    s=state["section"]
    context = kb_search.invoke({"query": s.search_query})
    msg = llm.invoke([
        SystemMessage(content="Write this one section in under 80 words, only from the context. End with the source file name in brackets."),
        HumanMessage(content=f"Section: {s.title}\n\nContext:\n{context}"),
    ])
    
    return {"completed": [f"## {s.title}\n{msg.content}"]}

def synthesizer(state: BriefState) -> BriefState:
    body = "\n\n".join(state["completed"])
    intro = llm.invoke([
        SystemMessage(content="Write a two-sentence welcome introduction for this brief."),
        HumanMessage(content=body),
    ])
    return {"brief": f"{intro.content}\n\n{body}"}

def build_graph():
  g= StateGraph(BriefState)
  g.add_node("orchestrator", orchestrator)
  g.add_node("worker", worker)
  g.add_node("synthesizer", synthesizer)

  g.add_edge(START, "orchestrator")
  g.add_conditional_edges("orchestrator", assign_workers, ["worker"])
  g.add_edge("worker", "synthesizer")
  g.add_edge("synthesizer", END)
  
  return g.compile()

def main():
    graph = build_graph()
    #if "--mermaid" in sys.argv:
    print(graph.get_graph().draw_mermaid())
    from pathlib import Path
    png = graph.get_graph(xray=1).draw_mermaid_png()
    Path("orcchestrator.png").write_bytes(png)
        
    enable_langsmith_tracing(Settings.LANGCHAIN_PROJECT)
    #result = graph.invoke({"task": "Write a first-week brief for a new hire who will work remotely and travel to clients."})
    result = graph.invoke({"task": "Write a first-week brief for a new hire"})
    print(result["brief"])

if __name__ == "__main__":
  main()
