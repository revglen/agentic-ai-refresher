from langchain_core.messages import HumanMessage, SystemMessage
from mcp.server.fastmcp import FastMCP

from config import Settings
from model_backend import get_chat_model
from rag import build_retriever_tool

mcp=FastMCP("refresher-tools")
_retriever_tool = build_retriever_tool(embedding_provider=Settings.PROVIDER)

@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '23 * 47' or '(3 + 5) / 2'."""

    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains characters that are not allowed."

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"

@mcp.tool()
def get_capital(country: str) -> str:
    """Determine the capital of a country.
       The country is passed to this method which then determine it's capital
    """
    provider = Settings.PROVIDER
    model_name = Settings.MODEL_NAME
    system_prompt = (
        "You are a geography assistant and know the name of all countries and their capital. \n"
        "You will give me given a country, you will need to determie it's capital. \n"
        "If there are more than one capital then you provide only one. \n"
        "Do not hallucinate and do not assume. If do not know the answer, please say so."
    )
    user_prompt = f"What is the capital of {country}"
    
    llm = get_chat_model(provider=provider, model_name=model_name)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return response.content

@mcp.tool()
def knowledge_base_search(query: str) -> str:
    """Search the internal sample knowledge base (remote work policy, expense
        policy, onboarding guide) for a passage relevant to the query."""
    return _retriever_tool.invoke({"query": query})

if __name__ == "__main__":
  mcp.run(transport="stdio")