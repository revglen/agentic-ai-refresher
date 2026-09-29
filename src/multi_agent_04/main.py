import argparse
from langchain_core.messages import HumanMessage

from config import Settings
from model_backend import enable_langsmith_tracing
from agents import mountain_agent
from supervisor import build_supervisor

#QUESTION = "What is the home office stipend, what is it times 4, and what is the capital of Mongolia?"
#QUESTION = "What is the home office stipend?"
QUESTION="Tell more about the Himalayas"

def run_supervisor(question: str, mermaid: bool=True):
    from supervisor import build_graph
    graph = build_graph()
    
    print(graph.get_graph().draw_mermaid())
    
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(result["messages"][-1].content)

def run_hierarchical(question: str, mermaid: bool):
    from hierarchical import build_graph
    graph = build_graph()

    mountain_alone = build_supervisor(
            members={
                "mountain_agent": mountain_agent(),
            },
            descriptions={
                "mountain_agent": "Returns mountain details based on range and user query",
            },
            name="mountain_along",
        )
    result = graph.invoke(
            {"messages": [HumanMessage(content="Tell me about the Western Ghats")]},
            {"configurable": {"thread_id": "main-awarm-deml"}},
    )
    
    print(result["messages"][-1].content)
    
    if mermaid == True:
        #print(graph.get_graph(xray=1).draw_mermaid())   
        from pathlib import Path
        png = graph.get_graph(xray=1).draw_mermaid_png()
        Path("hierarchical.png").write_bytes(png)
        return
        
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(result["messages"][-1].content)

def run_swarm(question: str, mermaid: bool):
    from handoffs_swarm import build_graph
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(checkpointer=InMemorySaver())
    #print(graph.get_graph().draw_mermaid())

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": "main-awarm-deml"}},
    )

    print(result["messages"][-1].content)

def run_tools(question: str, mermaid: bool):
    from agents_as_tools import build_graph
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())
    
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(result["messages"][-1].content)

PATTERNS = {
    "supervisor": run_supervisor,
    "hierarchical": run_hierarchical,
    "swarm": run_swarm,
    "tools": run_tools,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", nargs="?", default="tools", choices=[*PATTERNS, "all"])
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--mermaid", default=False, action="store_true", help="print the graph instead of running it")
    args = parser.parse_args()

    enable_langsmith_tracing(Settings.LANGCHAIN_PROJECT)

    for name, fn in PATTERNS.items():
        if args.pattern in (name, "all"):
            print(f"\n==================== {name} ====================")
            fn(args.question, args.mermaid)

if __name__ == "__main__":
    main()