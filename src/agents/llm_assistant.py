from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


# CONSTANTS
from configs import LLM_CONFIG


model = OllamaLLM(model=LLM_CONFIG["MODEL"])


def llm_assistant(messages, user_query, retrieved_info="No info"):
    if len(messages) == 0:
        messages = [
            ("system", LLM_CONFIG["SYSTEM_PROMPT"]),
            ("human", LLM_CONFIG["USER_PROMPT"].format(user_query=user_query, retrieved_info=retrieved_info)),
        ]
    else:
        messages.append(
            ("human", LLM_CONFIG["USER_PROMPT"].format(user_query=user_query, retrieved_info=retrieved_info))
        )
    prompt = ChatPromptTemplate(messages)

    chain = prompt | model
    result = chain.invoke({})
    messages.append(("assistant", result))
    return messages
