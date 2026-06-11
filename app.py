import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.title("Chat with your PDF")
st.caption("Powered by Ollama + LangChain + ChromaDB")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever()
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context below.\n\nContext: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

question = st.chat_input("Ask anything about the book...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(question)
            context = "\n".join([d.page_content for d in docs])
            messages = prompt.format_messages(
                context=context,
                chat_history=st.session_state.chat_history,
                question=question
            )
            answer = llm.invoke(messages).content
            st.write(answer)

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))