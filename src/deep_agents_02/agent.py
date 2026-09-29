import os
import sys

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings
from model_backend import enable_langsmith_tracing, get_chat_model
from rag import build_retriever_tool

from dotenv import load_dotenv
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
    agent = create_deep_agent(
               model=llm,
               tools=tools,
               system_prompt=(
                   "You are a carefull assistant. Use tools to determine how to solve the user's queries. \n" \
                   "Anything assocated with the HR policies, use the retriever_tool tool.\n"
                   "Anything asscoated with capital of a country use the get_capital tool.\n"
                   "Anything assocated with number calculation, use the calculator tool\n"
                   "For all other queries, are not supported and you say so explicitly. "
                   "Do not bring in any of youe knowledge and you will depends on the tools only.\n"
                   "Do not hallucinate and do not assume. If you do not know the answer say so explicily." 
               )
            )

    return agent

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