from sentence_transformers import SentenceTransformer

# 1. Load the model
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B", trust_remote_code=True, device="cuda", load_in_8bit=True
)

# 2. Prepare your texts
queries = ["What is the capital of France?", "How does a transformer model work?"]
documents = [
    "Paris is the capital and most populous city of France.",
    "The transformer model architecture is based on self-attention mechanisms.",
]

# 3. It's recommended to use a prompt for queries for better performance
query_embeddings = model.encode(queries, prompt_name="query")

# For documents, you can encode them directly
document_embeddings = model.encode(documents)

print("Shape of query embeddings:", query_embeddings.shape)
print("Shape of document embeddings:", document_embeddings.shape)

# You can now use these embeddings for tasks like semantic search
# For example, calculating cosine similarity between a query and documents
import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


similarity = cosine_similarity(query_embeddings[0], document_embeddings[0])
print(f"Similarity between the first query and the first document: {similarity}")
