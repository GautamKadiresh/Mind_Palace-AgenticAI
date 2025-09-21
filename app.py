from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


from agents.llm_assistant import llm_assistant
from agents.vectordb_retriever import retriever
from agents.vectordb_loader import vectordb_loader


class MessageClassifier(BaseModel):
    next_action: Literal["load", "router", "retrieve", "chat", "exit"] = Field(
        ...,
        description="Classify if the next action is to intiate vector db document load or retrieve infor from vector db or chat or exit.",
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
    print("\n\n-------------------------------")
    user_query = input("Ask your question (q to quit):\n")
    print("-------------------------------")
    if user_query == "q":
        print("Bye")
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
    print("Mind Palace:")
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
    print("Hello from mind-palace!\n\nPlease wait while the database is getting ready...\n")
    state = {"messages": [], "user_query": None, "retrieved_info": None, "next_action": "load"}

    state = graph.invoke(state)


if __name__ == "__main__":
    main()
