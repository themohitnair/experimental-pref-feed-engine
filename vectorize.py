from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
import numpy as np
from transformers import BitsAndBytesConfig

model_id = "Qwen/Qwen3-Embedding-8B"

# 1. Quantization config
bnb_config = BitsAndBytesConfig(load_in_8bit=True)  # or load_in_4bit=True

# 2. Build Transformer module (with quantization)
word_embedding_model = models.Transformer(
    model_id, trust_remote_code=True, device="cuda", quantization_config=bnb_config
)

# 3. Add pooling layer (to get sentence embeddings)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# 4. Wrap into SentenceTransformer
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 5. Test with queries + docs
queries = ["What is the capital of France?", "How does a transformer model work?"]
documents = [
    "Paris is the capital and most populous city of France.",
    "The transformer model architecture is based on self-attention mechanisms.",
]

query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

print("Shape of query embeddings:", query_embeddings.shape)
print("Shape of document embeddings:", document_embeddings.shape)


# 6. Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


similarity = cosine_similarity(query_embeddings[0], document_embeddings[0])
print(f"Similarity between the first query and the first document: {similarity}")
