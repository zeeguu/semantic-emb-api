import nltk
from app.semantic_vector import semantic_embedding_model

nltk.download("punkt")
api_model = semantic_embedding_model
if api_model.model is not None:
    print("Model downloaded and loaded!")
