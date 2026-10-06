import operator
import sys
from collections import Counter
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel

from config import Settings
from rag import build_retriever_tool
from model_backend import default_llm, enable_langsmith_tracing

llm=default_llm()
search = build_retriever_tool()

# ---------------------------- Sectioning ----------------------------
class SectionState(TypedDict, total=False):
    request: str
    findings: Annotated[list[str], operator.add]
    verdict: str

def make_checker(aspect: str, query: str):
    def check(state: SectionState) -> SectionState:
        context = search.invoke({"query": query})
        msg = llm.invoke([
            SystemMessage(content="You check only the {aspect}. Use only the context. Two senetences max."),
            HumanMessage(content=f"Context:\b{context}\n\nRequest:\n{state['request']}"),
        ])

        return {"findings": [f"{aspect.upper()}: {msg.content}"]}
    return check

def combine(state: SectionState) -> SectionState:
    msg = llm.invoke([
        SystemMessage(content="Combine the findings into one verdict: allowed, allowed with conditions, or not allowed. Then list the conditions."),
        HumanMessage(content="\n\n".join(state["findings"])),
    ])

    return {"verdict": msg.content}

def build_sectioning_graph():
    g=StateGraph(SectionState)

    checkers = {
        "expense_check": make_checker("expense", "meal cap alcohol approval threshold"),
        "remote_check": make_checker("remote work", "remote days per week manager approval"),
        "security_check": make_checker("security", "VPN encryption remote devices"),
    }

    for name, fn in checkers.items():
        g.add_node(name, fn)
        g.add_edge(START, name)

    g.add_node("combine", combine)
    g.add_edge(list(checkers), "combine")
    g.add_edge("combine", END)

    return g.compile()

# ------------------------------ Voting ------------------------------
class Vote(BaseModel):
    reimbursable: Literal["yes", "no"]

class VoteState(TypedDict, total=False):
    claim: str
    votes: Annotated[list[str], operator.add]
    result: str

def make_judge(persona: str):
    def judge(state: VoteState) -> VoteState:
        context=search.invoke({"query": state["claim"]})
        vote=llm.with_structured_output(Vote).invoke([
        SystemMessage(content=f"You are {persona}. Decide from the policy context only."),
                HumanMessage(content=f"Context:\n{context}\n\nIs this claim reimbursable? {state['claim']}"),
        ])
    
        return {"votes":[vote.reimbursable]}
    return judge

def tally(state: VoteState) -> VoteState:
    counts = Counter(state["votes"])
    winner, n = counts.most_common(1)[0]
    return {"result": f"{winner} ({n}/{len(state['votes'])} votes)"}

def build_voting_graph():
    g = StateGraph(VoteState)

    judges = {
        "judge_strict": make_judge("a strict finance auditor"),
        "judge_lenient": make_judge("a lenient line manager"),
        "judge_neutral": make_judge("a neutral HR officer"),
    }
    
    for name, fn in judges.items():
        g.add_node(name, fn)
        g.add_edge(START, name)
    
    g.add_node("tally", tally)
    
    g.add_edge(list(judges), "tally")
    g.add_edge("tally", END)
    
    return g.compile()

# ------------------------------ Main ------------------------------

def main():
    sectioning = build_sectioning_graph()
    if "--mermaid" in sys.argv:
        print(sectioning.get_graph().draw_mermaid())
        #print(voting.get_graph().draw_mermaid())
        return
    enable_langsmith_tracing(Settings.LANGCHAIN_PROJECT)

    r = sectioning.invoke({"request": "Can I work from my home office 4 days a week and expense a 55 dinner with wine on a client trip?"})
    print("SECTIONING\n" + "\n".join(r["findings"]) + "\n\nVERDICT\n" + r["verdict"])

    voting = build_voting_graph()
    r = voting.invoke({"claim": "A 35 team lunch while travelling, itemized receipt attached, submitted after 10 days."})
    print(f"\nVOTING: {r['votes']} -> {r['result']}")

if __name__ == "__main__":
    main()