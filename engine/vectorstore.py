from typing import List, Any, Tuple
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class MilvusVectorStore(VectorStore):
    """Milvus를 LangChain VectorStore 인터페이스로 래핑"""

    def __init__(self, client, collection_name: str, embeddings: Embeddings):
        self.client = client
        self.collection_name = collection_name
        self._embeddings = embeddings

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        pass

    @classmethod
    def from_texts(cls, texts: List[str], embedding, **kwargs):
        raise NotImplementedError("Use __init__ directly with existing Milvus collection")

    def similarity_search(self, query: str, k: int = 3, **kwargs: Any) -> List[Document]:
        query_embedding = self._embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=k,
            output_fields=["text_content"] if "text_search" in self.collection_name else []
        )
        documents = []
        for result in results[0]:
            content = (
                result['entity'].get('text_content', '')
                if "text_search" in self.collection_name
                else f"Image ID: {result['id']}"
            )
            doc = Document(
                page_content=content,
                metadata={"id": result['id'], "similarity_score": float(result['distance'])}
            )
            documents.append(doc)
        return documents

    def similarity_search_with_score(self, query: str, k: int = 3, **kwargs) -> List[Tuple]:
        docs = self.similarity_search(query, k, **kwargs)
        return [(doc, doc.metadata['similarity_score']) for doc in docs]
