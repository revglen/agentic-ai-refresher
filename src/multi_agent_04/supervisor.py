from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from model_backend import default_llm
from agents import geo_agent, math_agent, policy_agent

MAX_STEPS=6

class SupervisorState(MessagesState):
   steps: int

def build_supervisor(members: dict,
                     descriptions: dict[str, str], 
                     name:str="supervisor"):
    options=[*members, "FINISH"]
    Route = create_model("Route",
        next=(
            Literal[tuple(options)], 
            Field(description="Who acts next, or FINISH when the request is fully answered.")),
    )

    roaster="\n".join(f"- {n}: {d}" for n,d in descriptions.items())
    router = default_llm().with_structured_output(Route)
    llm = default_llm()

    def supervisor(state: SupervisorState) -> Command:
        steps = state.get("steps", 0)
        if steps >= MAX_STEPS:
            return Command(goto="respond")
        
        decision=router.invoke([
            SystemMessage(content=(
            f"You manage these workers:\n{roaster}\n\n"
                        "Read the conversation. Pick the worker for the next unanswered part of the "
                        "user's request. Choose FINISH once every part has a worker's answer."
            )),
            *state["messages"],
        ])

        goto = "respond" if decision.next == "FINISH" else decision.next
        return Command(goto=goto, update={"steps": steps+1})

    def make_worker_node(member_name, runnable):
        print("Type is", type(runnable))
        def node(state: SupervisorState) -> Command:
            result = runnable.invoke({"messages": state["messages"]})
            reply = result["messages"][-1].content

            return Command(
                goto="supervisor",
                update={"messages": [AIMessage(content=f"[{member_name}] {reply}", name=member_name)]},
            )
            
        return node

    def respond(state: SupervisorState) -> dict:
        msg = llm.invoke([
            SystemMessage(content="Combine the workers' answers into one reply to the user's original request. Use only what the workers said."),
                    *state["messages"],
                ])

        print("Graph is ", name)
        return {"messages": [AIMessage(content=msg.content, name=name)]}

    g=StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor, destinations=tuple([*members, "respond"]))
    for member_name, runnable in members.items():
        print(f"Member Name: {member_name}" )
        g.add_node(member_name, make_worker_node(member_name, runnable), destinations=("supervisor",))
    g.add_node("respond", respond)

    g.add_edge(START, "supervisor")
    g.add_edge("respond", END)
    
    return g.compile(name=name)

def build_graph():
   return build_supervisor(
      members={
        "policy_agent": policy_agent(),
        "math_agent": math_agent(),
        "geo_agent": geo_agent(),
     },
     descriptions={
            "policy_agent": "company HR policy lookups (remote work, expenses, onboarding)",
            "math_agent": "arithmetic with a calculator",
            "geo_agent": "capital cities",
        },
    )
   