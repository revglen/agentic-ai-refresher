import sys
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from config import Settings
from rag import build_retriever_tool
from model_backend import default_llm, enable_langsmith_tracing

MAX_ROUNDS = 3

CRITERIA = """
1. Every number matches the context exactly.
3. Ends with 'Source: <file name>'.
3. Under 90 words.
"""

class Grade(BaseModel):
  grade: Literal["pass", "fail"]
  feedback: str = Field(description="If fail, the specific fixes needed. If pass, 'ok'.")

class LoopState(TypedDict, total=False):
  question: str
  context: str
  answer: str
  grade: str
  feedback: str
  rounds: int

llm = default_llm()
kb_search = build_retriever_tool()

def generate(state: LoopState) -> LoopState:
    context = state.get("context") or kb_search.invoke({"query": state["question"]})
    prompt = f"Context:\n{context}\n\nQuestion:\n{state['question']}\n\nCriteria:\n{CRITERIA}"

    if state.get("feedback"):
        prompt += f"\n\nYour previous answer:{state['answer']}\n\nFix this feedback:\n{state['feedback']}"

    msg = llm.invoke([
        SystemMessage(content="Answer the employee's question from the context, meeting every criterion."),
        HumanMessage(content=prompt),
    ])

    return {"context": context, "answer": msg.content, "rounds": state.get("rounds", 0) + 1}

def evaluate(state: LoopState) -> LoopState:
    result = llm.with_structured_output(Grade).invoke([
        SystemMessage(content=f"Grade the answer strictly against these criteris:\n {CRITERIA}"),
        HumanMessage(content=f"Context:\n{state['context']}\n\nQuestion: {state['question']}\n\nAnswer:\n{state['answer']}"),
    ])

    print(f"  round {state['rounds']}: {result.grade} | {result.feedback}")
    return {"grade": result.grade, "feedback": result.feedback}

def next_step(state: LoopState) -> str:
   if state["grade"] == "pass" or state["rounds"] >= MAX_ROUNDS:
     return END

   return "generate"

def build_graph():
  g=StateGraph(LoopState)
  g.add_node("generate", generate)
  g.add_node("evaluate", evaluate)
  
  g.add_edge(START, "generate")
  g.add_edge("generate", "evaluate")
  g.add_conditional_edges("evaluate", next_step, ["generate", END])

  return g.compile()

def main():
    graph=build_graph()
    if "--mermaid" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        return

    enable_langsmith_tracing(Settings.LANGCHAIN_PROJECT)
    result = graph.invoke({"question": "Can I fly premium economy on a 5 hour flight, and who approves a 150 hotel bill?"})
    print(f"\nFinal ({result['grade']} after {result['rounds']} rounds):\n{result['answer']}")

if __name__ == "__main__":
   main()