from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings

class WatsonxAIEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "IBM watsonx.ai Embeddings"
    description = "Generate embeddings using watsonx.ai models."
    name = "IBMwatsonxEmbeddings"
    
    def build_embeddings(self) -> Embeddings:
        pass