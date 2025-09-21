from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


# CONSTANTS
from constants import LLM_ASSISTANT, SYSTEM_PROMPT, USER_PROMPT


model = OllamaLLM(model=LLM_ASSISTANT)


def llm_assistant(messages, user_query, retrieved_info="No info"):
    if len(messages) == 0:
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT.format(user_query=user_query, retrieved_info=retrieved_info)),
        ]
    else:
        messages.append(("human", USER_PROMPT.format(user_query=user_query, retrieved_info=retrieved_info)))
    prompt = ChatPromptTemplate(messages)

    chain = prompt | model
    result = chain.invoke({})
    messages.append(("assistant", result))
    return messages


# print("\n\n-------------------------------")
# user_query = input("Ask your question (q to quit):\n")
# print("-------------------------------")
# chat_history = llm_assistant(messages=[],user_query=user_query,retrieved_info='No info')
# print('#',chat_history[-1][1],'#')
