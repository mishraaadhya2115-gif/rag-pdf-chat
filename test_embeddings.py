from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

text = "The cat sat on the mat"
result = embeddings.embed_query(text)

print("Number of dimensions:", len(result))
print("First 5 numbers:", result[:5])