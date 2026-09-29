from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.types import Command

from agents import calculator, get_capital
from rag import build_retriever_tool
from model_backend import default_llm

AGENTS = ["triage_agent", "policy_agent", "numbers_agent"]

class SwarmState(MessagesState):
    active_agent: str

def make_handoff_tool(target: str, description: str):
    @tool(f"transfer_to_{target}", description=description)
    def handoff(runtime: ToolRuntime) -> Command:
        done = ToolMessage(content=f"Trandfered to {target}.", 
             tool_call_id=runtime.tool_call_id)
        return Command(
            goto=target,
            graph=Command.PARENT,
            update={
                    "messages":
                    [*runtime.state["messages"], done], 
                    "active_agent": target
                }
        )

    return handoff

to_policy = make_handoff_tool("policy_agent", "Hand over to the HR policy expert.")
to_numbers = make_handoff_tool("numbers_agent", "Hand over for arithmetic or capital cities.")
to_triage = make_handoff_tool("triage_agent", "Hand back when the request is outside your expertise.")

def build_graph(checkpointer=None):
    llm=default_llm()
    triage = create_agent(
        llm,
        tools=[to_policy, to_numbers], name="triage_agent",
        system_prompt="you greet the user and immediately hand over to the right specialist. Never answer yourself."
    )
    
    policy = create_agent(
            llm, 
            tools=[build_retriever_tool(), to_numbers, to_triage], name="policy_agent",
            system_prompt=(
                "You are the HR policy expert, Search the knowledge base and cite the source."
                "If the user also needs a calculation, state the figures you found , then hand over to number_agents."
            )
    )

    numbers = create_agent(
        llm,
        tools=[calculator, get_capital, to_policy, to_triage], name="numbers_agent",
        system_prompt=(
            "You do arithmetic with the calculator and capitals cities with get_capital. "
            "Use figures already in the conversation. Hand over to policy_agenr if you need a policy fact."
        )
    )

    g=StateGraph(SwarmState)

    g.add_node("triage_agent", triage, destinations=("policy_agent", "numbers_agent"))
    g.add_node("policy_agent", policy, destinations=("numbers_agent", "triage_agent"))
    g.add_node("numbers_agent", numbers, destinations=("policy_agent", "triage_agent"))
    g.add_conditional_edges(START, lambda s: s.get("active_agent") or "triage_agent", AGENTS)
    for name in AGENTS:
      g.add_edge(name, END)

    return g.compile(checkpointer=checkpointer)