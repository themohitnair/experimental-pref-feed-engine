from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load quantized HF model
model_id = "Qwen/Qwen3-Embedding-8B"

hf_model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    load_in_8bit=True,  # quantization happens here
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 2. Wrap into SentenceTransformer
model = SentenceTransformer(modules=[hf_model], tokenizer=tokenizer)

# 3. Encode queries/documents
queries = ["What is the capital of France?", "How does a transformer model work?"]
documents = [
    "Paris is the capital and most populous city of France.",
    "The transformer model architecture is based on self-attention mechanisms.",
]

query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

print("Shape of query embeddings:", query_embeddings.shape)
print("Shape of document embeddings:", document_embeddings.shape)


# 4. Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


similarity = cosine_similarity(query_embeddings[0], document_embeddings[0])
print(f"Similarity between the first query and the first document: {similarity}")
