from .semantic_vector_model import SemanticVectorModel
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import os
import sys

MODULE_PATH = os.path.dirname(__file__)
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
SENTENCE_MODEL_PATH = os.path.join(MODULE_PATH, "binaries", MODEL_NAME)


class SentenceXMLRoberta(SemanticVectorModel):
    def __init__(self) -> None:
        super().__init__()
        if os.path.exists(SENTENCE_MODEL_PATH):
            print(f"Saved model found! Loading Model '{MODEL_NAME}'")
            self.model_name = SENTENCE_MODEL_PATH
            self.model = SentenceTransformer(SENTENCE_MODEL_PATH)
        else:
            print(f"Model not found: Downloading model '{MODEL_NAME}'!")
            self.model_name = MODEL_NAME
            self.model = SentenceTransformer(MODEL_NAME)
            self.model.save(SENTENCE_MODEL_PATH)
        self.model.compile()

    def get_vector(self, text: str, language: str = "english") -> list:
            try:
                return (
                    self.model.encode(sent_tokenize(text, language=language))
                    .mean(axis=0)
                    .tolist()
                )
            except LookupError:
                # Use the default (English)
                print("# Warning, using default language (English) to senticize.", file=sys.stderr)
                return self.model.encode(sent_tokenize(text)).mean(axis=0).tolist()

    def get_model_name(self) -> str:
        return self.model_name
