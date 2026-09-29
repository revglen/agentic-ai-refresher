import os
import sys
import asyncio

from deepagents import create_deep_agent

from config import Settings
from model_backend import enable_langsmith_tracing, get_chat_model
from rag import build_retriever_tool

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")

async def agent_setup():
    enable_langsmith_tracing("refresher-mcp-server-agent")
    provider = Settings.PROVIDER
    model_name = Settings.MODEL_NAME
 
    client=MultiServerMCPClient(
       {
          "refresher-tools":{
            "command": sys.executable,
            "args": [SERVER_SCRIPT],
            "transport": "stdio",
          }
       }
    )

    tools = await client.get_tools()
    print(tools)
    llm=get_chat_model(provider=provider, model_name=model_name)
    agent=create_agent(llm, tools)

    return agent    

async def run(agent, query):
    result = await agent.ainvoke({"messages": [("human", query)]})
    return result["messages"][-1].content

async def async_main():
    agent = await agent_setup()

    query="What is the remote work policy?"
    result = await run(agent, query)
    print(result)
        
    query="India"
    result = await run(agent, query)
    print(result)
        
    query="2*4+2"
    result = await run(agent, query)
    print(result)            

def main():
    enable_langsmith_tracing(Settings.LANGCHAIN_PROJECT)
    asyncio.run(async_main())    

if __name__ == "__main__":
    main()