from agents import (
        geo_agent, 
        math_agent, 
        policy_agent, 
        writer_agent,
        mountain_agent
    )
from supervisor import build_supervisor

def build_graph():
    knowledge_team = build_supervisor(
        members={
                    "policy_agent": policy_agent(),
                    "writer_agent": writer_agent()
                },
        descriptions={
            "policy_agent": "looks up HR policy facts",
            "writer_agent": "rewrites the facts found into friendly prose",
        },
        name="knowledge_team",
    )

    
    useful_agents = build_supervisor(
        members={
                "math_agent": math_agent(),
                "geo_agent": geo_agent()
        },
        descriptions={
                "math_agent": "arithmetic", 
                "geo_agent": "capital cities"
        },
        name="members_team",    
    )

    mountain_team = build_supervisor(
        members={
            "mountain_agent": mountain_agent(),
            #"writer_agent": writer_agent(),
        },
        descriptions={
            "mountain_agent": "Returns mountain details based on range and user query",
            #"writer_agent": "rewrites the facts found into friendly prose",
        },
        name="mountain_team",
    )

    return build_supervisor(
        members={
            "knowledge_team": knowledge_team, 
            "useful_agents": useful_agents,
            "mountain_team": mountain_team
        },
        descriptions={
            "knowledge_team": "anything about company policy, written up for an employee",
            "numbers_team": "calculations and capital cities",
            "mountain_team": "mountain details"
        },
        name="top_supervisor",
    )
