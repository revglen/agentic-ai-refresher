import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from create_agent_01.config import Settings
from create_agent_01.model_backend import enable_langsmith_tracing, get_chat_model
from create_agent_01.rag import build_retriever_tool

path1=os.path.dirname(__file__)
path2=os.path.dirname(__file__) + "/.."
path3=os.path.dirname(__file__) + "/../.."
sys.path.insert(0, os.path.join(os.path.dirname(__file__), path1, path2, path3))
print(sys.path)

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '23 * 47' or '(3 + 5) / 2'."""

    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains characters that are not allowed."

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
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

def build_agent():
    provider = Settings.PROVIDER
    model_name = Settings.MODEL_NAME

    llm = get_chat_model(provider=provider, model_name=model_name)
    retriever_tool = build_retriever_tool(embedding_provider=provider)
    tools = [calculator, retriever_tool, get_capital]
    return create_agent(llm, tools)

def main():

    enable_langsmith_tracing(Settings.LANGCHAIN_PROJECT)

    agent = build_agent()

    query="What is the remote work policy?"
    result = agent.invoke({"messages": [("human", query)]})
    print(result["messages"][-1].content)

    query="India"
    result = agent.invoke({"messages": [("human", query)]})
    print(result["messages"][-1].content)

    query="2*4+2"
    result = agent.invoke({"messages": [("human", query)]})
    print(result["messages"][-1].content)
    

if __name__ == "__main__":
    main()