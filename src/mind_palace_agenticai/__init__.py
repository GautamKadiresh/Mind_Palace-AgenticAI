from typing import Annotated
from typing_extensions import TypedDict

import os

os.environ["USER_AGENT"] = "Mind_Palace-AgenticAI"
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# import custom agents
from agents.llm_assistant import llm_assistant
from agents.vectordb_retriever import retriever
from agents.vectordb_loader import vectordb_loader

# terminal output formatting constants
from constants import (
    BOLD_RED,
    BOLD_GREEN,
    BOLD_YELLOW,
    BOLD_MAGENTA,
    DIM_CYAN,
    RESET_FONT,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str | None
    retrieved_info: str | None
    next_action: str | None


def ingestor_agent(state: State):
    if state["next_action"] == "load":
        vectordb_loader()
        state["next_action"] = "retrieve"
        return state
    else:
        return state


def router(state: State):
    print(f"\n{BOLD_RED}____________________________________________________________________{RESET_FONT}")
    print(f"{BOLD_GREEN}Ask your question (q to quit):{RESET_FONT}")
    user_query = input()
    print(f"\n{BOLD_RED}____________________________________________________________________{RESET_FONT}")
    if user_query == "q":
        print(f"\n{BOLD_YELLOW}BYE!!{RESET_FONT}\n")
        return {"messages": state["messages"], "user_query": None, "retrieved_info": None, "next_action": "exit"}
    else:
        return {
            "messages": state["messages"],
            "user_query": user_query,
            "retrieved_info": None,
            "next_action": "retrieve",
        }


def retriever_agent(state: State):
    retrieved_info = retriever.invoke(state["user_query"])
    retrieved_info = [str(i).replace("{", "{{").replace("}", "}}") for i in retrieved_info]
    return {
        "messages": state["messages"],
        "user_query": state["user_query"],
        "retrieved_info": retrieved_info,
        "next_action": "chat",
    }


def assistant_agent(state: State):
    chat_history = llm_assistant(
        messages=state["messages"], user_query=state["user_query"], retrieved_info=state["retrieved_info"]
    )
    print(f"{BOLD_MAGENTA}Mind Palace:{RESET_FONT}")
    print(chat_history[-1][1])
    return {"messages": chat_history, "user_query": None, "retrieved_info": None, "next_action": "router"}


graph_builder = StateGraph(State)


graph_builder.add_node("ingestor_agent", ingestor_agent)
graph_builder.add_node("router", router)
graph_builder.add_node("retriever_agent", retriever_agent)
graph_builder.add_node("assistant_agent", assistant_agent)


graph_builder.add_edge(START, "ingestor_agent")
graph_builder.add_edge("ingestor_agent", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state["next_action"],
    {"exit": END, "retrieve": "retriever_agent"},  # mapping: condition -> next node
)
graph_builder.add_edge("retriever_agent", "assistant_agent")
graph_builder.add_edge("assistant_agent", "router")


graph = graph_builder.compile()


def main():
    print(f"{BOLD_MAGENTA}Hello from mind-palace!{RESET_FONT}\n")

    state = {"messages": [], "user_query": None, "retrieved_info": None, "next_action": "load"}
    try:
        state = graph.invoke(state)
    except Exception as e:
        print(
            f"{BOLD_RED}Error: An unexpected error occurred. Please ensure Ollama is running and accessible.\nDetails: {str(e)}{RESET_FONT}"
        )
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
