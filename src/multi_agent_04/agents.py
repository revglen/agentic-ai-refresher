from langchain.agents import create_agent

from model_backend import default_llm
from rag import build_retriever_tool
from tools import calculator, get_capital,mounatin_details

def policy_agent(extra_tools=(), name="policy_agent"):
    return create_agent(
        default_llm(),
        tools=[build_retriever_tool(), *extra_tools],
        system_prompt=(
            "you are the HR policy expert. Always search the knowledge base, answer only "
            "from it, and cite the source file. Quote numbers exactly."),
        name=name,
    )

def math_agent(extra_tools=(), name="math_agent"):
    return create_agent(
        default_llm(),
        tools=[calculator, *extra_tools],
        system_prompt=(
            "You do arithmetic."
            "Always use the calculator tool; never compute in your head."
        ),
        name=name,
    )

def geo_agent(extra_tools=(), name="geo_agent"):
    return create_agent(
        default_llm(),
        tools=[get_capital, *extra_tools],
        system_prompt=(
            "You do arithmetic."
            "Always use the calculator tool; never compute in your head."
        ),
        name=name,
    )

def mountain_agent(extra_tools=(), name="mountain_agent"):
    return create_agent(
        default_llm(),
        tools=[mounatin_details, *extra_tools],
        system_prompt=(
            "You are mountain expert."
            "Always use the mounatin_details tool; never compute details on your own.\n"
            "Always follow instructions and say if so if you do not know"
        ),
        name=name,
    )

def writer_agent(name="writer_agent"):
    return create_agent(
        default_llm(),
        tools=[],
        system_prompt="You rewrite material into clear, short, employee-friendly prose. Keep every fact and number.",
        name=name,
    )
