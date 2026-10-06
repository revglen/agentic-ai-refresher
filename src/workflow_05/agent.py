import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.tools import tool

from config import Settings
from model_backend import enable_langsmith_tracing, get_chat_model
from rag import build_retriever_tool

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