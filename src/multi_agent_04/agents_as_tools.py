from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents import policy_agent, math_agent, geo_agent
from model_backend import default_llm

_policy = policy_agent()
_math = math_agent() 
_geo = geo_agent()

def _ask(agent, question: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content

@tool
def ask_policy_expert(question: str) -> str:
    """Ask the HR policy expert a self-contained question about remote work, expenses or onboarding."""
    return _ask(_policy, question)

@tool
def ask_math_expert(question: str) -> str:
  """Ask the math expert to compute something. Include every number it needs."""
  return _ask(_math, question)

@tool
def ask_geo_expert(question: str) -> str:
  """Ask the geography expert for a country's capital."""
  return _ask(_geo, question)

def build_graph():
    return create_agent(
        default_llm(),
        tools=[ask_policy_expert, ask_math_expert, ask_geo_expert],
        system_prompt=(
            "You coordinate specialists. Break the request into parts, ask the right expert "
            "for each (experts cannot see this conversation, so pass full context), then "
            "answer the user using only the experts' replies."
        ),
        name="orchestrator",
    )
