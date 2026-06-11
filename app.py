import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

load_dotenv()

st.title("Chat with your PDF 🤖")
st.caption("Powered by Groq + LangChain + ChromaDB")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context below.\n\nContext: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.db is None:
    with st.spinner("Reading and indexing your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        st.session_state.db = Chroma.from_documents(chunks, embedding=embeddings)
    st.success(f"Done! {len(chunks)} chunks indexed. Ask away!")

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

if st.session_state.db:
    question = st.chat_input("Ask anything about your PDF...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.db.as_retriever()
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
else:
    st.info("Please upload a PDF to get started!")