import numpy as np
import torch
from typing import List
from transformers import CLIPTokenizerFast, CLIPModel
from langchain_core.embeddings import Embeddings


class CLIPEmbeddings(Embeddings):
    """CLIP 모델을 LangChain Embeddings 인터페이스로 래핑"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.device = "cpu"
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            text_features = self.model.text_projection(outputs.pooler_output)
        text_features = text_features.detach().cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        return text_features.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
