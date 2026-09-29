from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config import Settings
from model_backend import default_llm
from rag import build_retriever_tool

_retriever_tool = build_retriever_tool(embedding_provider=Settings.PROVIDER)

tool()
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '23 * 47' or '(3 + 5) / 2'."""

    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains characters that are not allowed."

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"

tool()
def get_capital(country: str) -> str:
    """Determine the capital of a country.
       The country is passed to this method which then determine it's capital
    """
    system_prompt = (
        "You are a geography assistant and know the name of all countries and their capital. \n"
        "You will give me given a country, you will need to determie it's capital. \n"
        "If there are more than one capital then you provide only one. \n"
        "Do not hallucinate and do not assume. If do not know the answer, please say so."
    )
    user_prompt = f"What is the capital of {country}"
    
    response = default_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return response.content

tool()
def knowledge_base_search(query: str) -> str:
    """Search the internal sample knowledge base (remote work policy, expense
        policy, onboarding guide) for a passage relevant to the query."""
    return _retriever_tool.invoke({"query": query})

tool()
def mounatin_details(mountain_query: str) -> str:
    """Details of a montain details based on user query and mountain range will be returned"""

    system_prompt =(
        "You are mountain knowledge specialist with deep understanding of earth's mountain ranges. "
        "You have details of the peaks, highest range and hiking parts.\n"
        "The user has a query of this mountain range hence extract all information you can.\n"
        "If you do not know about the answer, say so explictly and do not guess"
    )

    user_prompt = (f"Please provide mountain details for the user query : {mountain_query}")

    response = default_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return response.content