import os
import hashlib
import glob
from typing import List, Tuple
from models import Chunk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from input_data import *
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
import torch
import gc
logger = logging.getLogger(__name__)

def flush():
    torch.mps.empty_cache()
    gc.collect()


def sigmoid(logits, temp=1.) -> torch.Tensor:
    return torch.sigmoid(logits / temp)


class Reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(HF_RERANKER_LOCAL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            HF_RERANKER_LOCAL_PATH,
            device_map='auto',
            torch_dtype=torch.bfloat16
        ).eval()


    def rerank_documents(self, query: str, documents: list[Chunk], numDocs:int = 5, batch_size: int = 8) -> list[Chunk]:
        pairs = [[query, doc.text] for doc in documents]
        all_probs = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            with torch.no_grad(): 
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.model.device)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
                probs = sigmoid(scores, temp=1.5).detach().cpu().numpy()
                all_probs.extend(probs)
            
        for doc, score in zip(documents, all_probs):
            doc.score = score

        sorted_docs = sorted(list(zip(documents, all_probs)), key=lambda x: x[1], reverse=True)[:numDocs]
        return [x[0] for x in sorted_docs]


class Retriever:
    """Простой ретривер с гибридным поиском"""

    def __init__(self, embed_model: str = "bge-m3", top_k: int = 10):
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.reranker = Reranker()
        self.faiss_store = None
        self.bm25_retriever = None
        self.basic_search_k: int = 20
        self.chunk_store = []
        self.cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
        self.top_k: int = top_k


    def _get_documents_hash(self, data_dir: str) -> str:
        import hashlib
        content = "".join(chunk.text for chunk in self.chunk_store)
        return hashlib.md5(content.encode()).hexdigest()


    def build(self, chunks: List[Chunk], use_cache: bool = True):
        """Строит индексы"""
        if not chunks:
            raise ValueError("Список chunks не может быть пустым")
        
        self.chunk_store = chunks
        
        self.chunk_by_id = {int(chunk.chunk_id): chunk for chunk in chunks}
        
        docs = [
            Document(page_content=c.text, metadata={"id": c.chunk_id, 'path': c.source_path}) for c in chunks
        ]

        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = self.basic_search_k
        
        if use_cache and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, "faiss_index")
            hash_path = os.path.join(self.cache_dir, "hash.txt")
            
            if os.path.exists(cache_path) and os.path.exists(hash_path):
                try:
                    with open(hash_path, "r") as f:
                        if f.read().strip() == self._get_documents_hash(self.cache_dir):
                            logger.info("Загружаем из кэша...")
                            self.faiss_store = FAISS.load_local(cache_path, self.embeddings, allow_dangerous_deserialization=True)
                            return
                except Exception as e:
                    logger.error(f"Ошибка при загрузке кэша: {e}")

        logger.info("Строим индексы...")
        
        self.faiss_store = FAISS.from_documents(docs, self.embeddings)
                
        if use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            try:
                self.faiss_store.save_local(cache_path)
                with open(hash_path, "w") as f:
                    f.write(self._get_documents_hash(self.cache_dir))
            except Exception as e:
                logger.error(f"Ошибка при сохранении кэша: {e}")

  


    def cast_documents_to_chunks(self, docs) -> list[Chunk]:
        return [Chunk(
            text=doc.page_content, source_path=doc.metadata['path'], chunk_id=doc.metadata['id'], image_paths=doc.metadata['images']
        ) for doc in docs]


    def search(self, query: str) -> List[Tuple[Chunk, float]]:
        """Гибридный поиск с расширенным контекстом"""
        if not self.faiss_store or not self.bm25_retriever:
            raise RuntimeError("Индексы не построены")
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.bm25_retriever, self.faiss_store.as_retriever(search_kwargs={'k': self.basic_search_k})
            ],
            weights=[0.5, 0.5]
        )
        retrieved_chunks = self.cast_documents_to_chunks(ensemble_retriever.invoke(query))
        logger.info(f'Найдено документов ретривером: {len(retrieved_chunks)}')
       
        # Ранжируем документы
        ranked_chunks = self.reranker.rerank_documents(query, retrieved_chunks, numDocs=self.top_k)
        logger.info(f'Получено документов после реранкинга: {len(ranked_chunks)}')
        
       
        return ranked_chunks

