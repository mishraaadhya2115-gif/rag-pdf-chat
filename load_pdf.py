from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("book.pdf")
pages = loader.load()

print("Total pages loaded:", len(pages))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

print("Total chunks created:", len(chunks))
print("\nExample chunk:")
print(chunks[0].page_content)
