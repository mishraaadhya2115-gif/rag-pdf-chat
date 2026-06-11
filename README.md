\# RAG PDF Chat 🤖



An AI-powered chatbot that lets you have conversations with any PDF document.

Ask questions in plain English and get intelligent answers with context from the document.



\## Demo

\[Add your demo video link here]



\## Features

\- Upload any PDF and chat with it instantly

\- Semantic search using vector embeddings

\- Remembers conversation history

\- Runs completely locally — no data sent to the cloud

\- Free to use (powered by Ollama)



\## Tech Stack

\- \*\*LangChain\*\* — AI application framework

\- \*\*ChromaDB\*\* — Vector database for semantic search

\- \*\*Ollama + LLaMA 3.2\*\* — Local AI model (free, runs on your machine)

\- \*\*Streamlit\*\* — Web interface

\- \*\*Python\*\* — Backend



\## How It Works

1\. PDF is loaded and split into small chunks

2\. Each chunk is converted into embeddings (numbers representing meaning)

3\. Embeddings are stored in ChromaDB

4\. When you ask a question, it finds the most relevant chunks

5\. LLaMA 3.2 reads those chunks and generates an answer



\## Installation



```bash

git clone https://github.com/mishraaadhya2115-gif/rag-pdf-chat.git

cd rag-pdf-chat

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



Install Ollama from https://ollama.com and run:

```bash

ollama pull llama3.2

ollama pull nomic-embed-text

```



\## Usage

```bash

streamlit run app.py

```



\## Author

Built by Aadhya Mishra

