from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever()
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context below.\n\nContext: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

chat_history = []

print("Chat with your PDF! Type 'exit' to quit.\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    messages = prompt.format_messages(
        context=context,
        chat_history=chat_history,
        question=question
    )
    answer = llm.invoke(messages).content
    print(f"\nAI: {answer}\n")
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))