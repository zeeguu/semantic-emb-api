from .semantic_vector_model import SemanticVectorModel
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import os
import sys

MODULE_PATH = os.path.dirname(__file__)
MODEL_NAME = "distiluse-base-multilingual-cased-v2"
SENTENCE_BERT_PATH = os.path.join(MODULE_PATH, "binaries", MODEL_NAME)


class SentenceBert(SemanticVectorModel):
    def __init__(self) -> None:
        super().__init__()
        # https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
        # The model is saved localy in the binaries folder.
        # The model encodes only up to 128 tokens. To encode large documents either
        # Use a Average / Pool Method
        # Loads model from internet:
        # self.model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
        if os.path.exists(SENTENCE_BERT_PATH):
            print(f"Saved model found! Loading Model '{MODEL_NAME}'")
            self.model_name = SENTENCE_BERT_PATH
            self.model = SentenceTransformer(SENTENCE_BERT_PATH)
        else:
            print(f"Model not found: Downloading model '{MODEL_NAME}'!")
            self.model_name = MODEL_NAME
            self.model = SentenceTransformer(MODEL_NAME)
            self.model.save(SENTENCE_BERT_PATH)
        self.model.compile()

    def get_vector(self, text: str, language: str = "english") -> list:
        try:
            sents_to_encode = sent_tokenize(text, language=language)
        except LookupError:
            # Use the default (English)
            sents_to_encode = sent_tokenize(text)
            print(
                "# Warning, using default language (English) to senticize.",
                file=sys.stderr,
            )
        return self.model.encode(sents_to_encode).mean(axis=0).tolist()

    def get_model_name(self) -> str:
        return self.model_name
