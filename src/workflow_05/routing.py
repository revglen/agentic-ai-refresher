import sys
from typing import Literal, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from agent import calculator, get_capital
from config import Settings
from rag import build_retriever_tool
from model_backend import default_llm, enable_langsmith_tracing

class Route(BaseModel):
    destination: Literal ["policy", "geography", "math", "unsupported"] = Field( description="policy: company HR policy. geography: a country or its capital. "
                    "math: an arithmetic expression. unsupported: anything else."
    )

class Expression(BaseModel):
    expression: str=Field(description="Arithmetic using only digits, + - * / ( ) and spaces.")

class RouteState(TypedDict, total=False):
    question: str
    destination: str
    answer: str

llm=default_llm()
kb_search=build_retriever_tool()

def structured(llm, schema, messages):
    msg = llm.bind(format=schema.model_json_schema()).invoke(messages)
    return schema.model_validate_json(msg.content)

def classifiy(state: RouteState) -> RouteState:
    # route = llm.with_structured_output(Route).invoke([
    #     SystemMessage(content="Classify the user's input."),
    #     HumanMessage(content=state["question"]),
    # ])

    route = structured(llm, Route, [
        SystemMessage(content="Classify the user's input."),
        HumanMessage(content=state["question"]),
    ])

    print(route)

    return {"destination": route.destination}

def policy(state: RouteState) -> RouteState:
    context = kb_search.invoke({"query": state["question"]})
    msg = llm.invoke([
            SystemMessage(content="Answer only from the context and cite the source file."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
        ])
    
    return {"answer": msg.content}

def geography(state: RouteState) -> RouteState:
    return {"answer": get_capital.invoke({"country": state["question"]})}

def math(state: RouteState) -> RouteState:
    expr = llm.with_structured_output(Expression).invoke([
        SystemMessage(content="Extract the arithmetic expression to evaluate."),
        HumanMessage(content=state["question"]),
    ])
    return {"answer": f"{expr.expression} = {calculator.invoke({'expression': expr.expression})}"}

def unsupported(state: RouteState) -> RouteState:
    return {"answer": "Sorry, I only handle company policy, capital cities and arithmetic."}

def pick_next_node(state):
    return state["destination"]

def build_graph():
  g=StateGraph(RouteState)
  
  g.add_node("classify", classifiy)
  for name, fn in [("policy", policy), ("geography", geography),("math", math), ("unsupported", unsupported)]:
    g.add_node(name, fn)
    g.add_edge(name, END)

  g.add_edge(START, "classify")
  g.add_conditional_edges("classify", pick_next_node,
                        ["policy", "geography", "math", "unsupported"])

  return g.compile()

def main():
    graph = build_graph()
    if "--mermaid" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        return
    enable_langsmith_tracing(Settings.LANGCHAIN_PROJECT)
    for q in ["What is the remote work policy?", "India", "2*4+2", "What's the weather in Dublin?"]:
        result = graph.invoke({"question": q})
        print(f"\n[{result['destination']}] {q}\n{result['answer']}")

if __name__ == "__main__":
    main()