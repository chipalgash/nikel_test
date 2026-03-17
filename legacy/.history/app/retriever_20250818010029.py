#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Гибридный поисковый движок с переранжированием
==============================================

Модуль для семантического и лексического поиска по корпусу документов
с использованием FAISS, BM25 и нейронного reranker'а.

Архитектура:
    1. Гибридный поиск: FAISS (семантика) + BM25 (лексика)
    2. Ансамблевое ранжирование с настраиваемыми весами
    3. Нейронное переранжирование топ результатов
    4. Кэширование индексов для быстрого перезапуска

Автор: AI Assistant
Дата: 2024
"""

import os
import hashlib
import logging
from typing import List, Tuple, Dict, Optional

import torch
import gc
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from models import Chunk
from input_data import HF_RERANKER_LOCAL_PATH

# Настройка логирования
logger = logging.getLogger(__name__)

# =============================================================================
# УТИЛИТЫ ДЛЯ РАБОТЫ С ПАМЯТЬЮ
# =============================================================================

def flush_gpu_memory() -> None:
    """
    Очищает память GPU для предотвращения переполнения.
    
    Поддерживает разные бэкенды: CUDA, MPS (Apple Silicon), CPU.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        logger.debug("🧹 GPU память очищена")
    except Exception as e:
        logger.warning(f"⚠️ Ошибка очистки GPU памяти: {e}")


def sigmoid(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Применяет сигмоидную функцию с температурой для калибровки оценок.
    
    Args:
        logits: Выходные логиты модели
        temperature: Параметр температуры для смягчения/обострения распределения
        
    Returns:
        torch.Tensor: Нормализованные вероятности
    """
    return torch.sigmoid(logits / temperature)

# =============================================================================
# КЛАСС ПЕРЕРАНЖИРОВЩИКА
# =============================================================================

class Reranker:
    """
    Нейронный переранжировщик для точного ранжирования результатов поиска.
    
    Использует предобученную модель BGE-reranker-v2-m3 для оценки
    релевантности пар "запрос-документ".
    
    Особенности:
        - Поддержка батчевой обработки
        - Автоматическое управление памятью GPU
        - Настраиваемая температура для калибровки оценок
    """
    
    def __init__(self, model_path: str = HF_RERANKER_LOCAL_PATH) -> None:
        """
        Инициализация reranker модели.
        
        Args:
            model_path: Путь к локальной модели BGE reranker
            
        Raises:
            RuntimeError: При ошибках загрузки модели
        """
        logger.info("🔄 Загрузка reranker модели...")
        
        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Загружаем модель с оптимизациями
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                device_map='auto',           # Автоматическое размещение на GPU/CPU
                torch_dtype=torch.bfloat16   # Эффективный тип данных
            ).eval()  # Режим инференса
            
            logger.info(f"✅ Reranker загружен: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки reranker: {e}")
            raise RuntimeError(f"Не удалось загрузить reranker: {e}")

    def rerank_documents(
        self, 
        query: str, 
        documents: List[Chunk], 
        num_docs: int = 5,
        batch_size: int = 8,
        temperature: float = 1.5
    ) -> List[Chunk]:
        """
        Переранжирует документы по релевантности к запросу.
        
        Args:
            query: Поисковый запрос
            documents: Список документов для ранжирования
            num_docs: Количество топ документов для возврата
            batch_size: Размер батча для обработки
            temperature: Температура для калибровки оценок
            
        Returns:
            List[Chunk]: Топ документы, отсортированные по релевантности
            
        Процесс:
            1. Формирование пар "запрос-документ"
            2. Батчевая обработка через нейронную сеть
            3. Применение температурной калибровки
            4. Сортировка по убыванию релевантности
        """
        if not documents:
            logger.warning("⚠️ Нет документов для переранжирования")
            return []
            
        logger.debug(f"🔄 Переранжирование {len(documents)} документов...")
        
        # Формируем пары для обработки
        query_doc_pairs = [[query, doc.text] for doc in documents]
        all_scores = []
        
        try:
            # Обрабатываем батчами для эффективности
            for i in range(0, len(query_doc_pairs), batch_size):
                batch = query_doc_pairs[i:i + batch_size]
                
                with torch.no_grad():
                    # Токенизация с ограничением длины
                    inputs = self.tokenizer(
                        batch, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='pt', 
                        max_length=512
                    ).to(self.model.device)
                    
                    # Получаем оценки релевантности
                    outputs = self.model(**inputs, return_dict=True)
                    logits = outputs.logits.view(-1).float()
                    
                    # Применяем температурную калибровку
                    scores = sigmoid(logits, temperature).detach().cpu().numpy()
                    all_scores.extend(scores)
                
                # Очищаем память после каждого батча
                flush_gpu_memory()
            
            # Добавляем оценки к документам
            for doc, score in zip(documents, all_scores):
                doc.score = float(score)
            
            # Сортируем по убыванию релевантности
            ranked_docs = sorted(
                documents, 
                key=lambda x: getattr(x, 'score', 0.0), 
                reverse=True
            )[:num_docs]
            
            logger.info(f"✅ Переранжировано в топ-{len(ranked_docs)} документов")
            
            # Логируем оценки для отладки
            if ranked_docs:
                top_score = ranked_docs[0].score
                avg_score = np.mean([doc.score for doc in ranked_docs])
                logger.debug(f"📊 Топ оценка: {top_score:.3f}, средняя: {avg_score:.3f}")
            
            return ranked_docs
            
        except Exception as e:
            logger.error(f"❌ Ошибка переранжирования: {e}")
            # Возвращаем исходные документы без ранжирования
            return documents[:num_docs]

# =============================================================================
# ОСНОВНОЙ КЛАСС ПОИСКОВОГО ДВИЖКА
# =============================================================================

class Retriever:
    """
    Гибридный поисковый движок с семантическим и лексическим поиском.
    
    Архитектура:
        - FAISS: Векторный поиск по эмбеддингам (семантика)
        - BM25: Статистический поиск по ключевым словам (лексика)
        - Ensemble: Комбинирование результатов с настраиваемыми весами
        - Reranker: Нейронное переранжирование финальных результатов
    
    Особенности:
        - Кэширование индексов для быстрого перезапуска
        - Настраиваемые параметры поиска
        - Автоматическое управление памятью
    """
    
    def __init__(
        self, 
        embed_model: str = "bge-m3", 
        top_k: int = 10,
        basic_search_k: int = 20
    ) -> None:
        """
        Инициализация гибридного retriever'а.
        
        Args:
            embed_model: Название модели эмбеддингов в Ollama
            top_k: Количество итоговых результатов
            basic_search_k: Количество кандидатов для каждого метода поиска
        """
        logger.info("🔧 Инициализация гибридного retriever'а...")
        
        # Настройки поиска
        self.top_k = top_k
        self.basic_search_k = basic_search_k
        
        # Пути для кэширования
        self.cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
        
        # Хранилища данных
        self.chunk_store: List[Chunk] = []
        self.chunk_by_id: Dict[int, Chunk] = {}
        
        # Компоненты поиска (инициализируются в build)
        self.faiss_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.reranker: Optional[Reranker] = None
        
        try:
            # Инициализация эмбеддингов
            self.embeddings = OllamaEmbeddings(model=embed_model)
            logger.info(f"✅ Эмбеддинги инициализированы: {embed_model}")
            
            # Инициализация reranker'а
            self.reranker = Reranker()
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации retriever'а: {e}")
            raise

    def _get_corpus_hash(self) -> str:
        """
        Вычисляет хэш корпуса для проверки изменений.
        
        Returns:
            str: MD5 хэш объединённого текста всех чанков
        """
        if not self.chunk_store:
            return ""
            
        # Объединяем весь текст корпуса
        combined_text = "".join(chunk.text for chunk in self.chunk_store)
        
        # Вычисляем MD5 хэш
        return hashlib.md5(combined_text.encode('utf-8')).hexdigest()

    def build(self, chunks: List[Chunk], use_cache: bool = True) -> None:
        """
        Строит все поисковые индексы на основе корпуса чанков.
        
        Args:
            chunks: Список чанков для индексирования
            use_cache: Использовать кэш если возможно
            
        Raises:
            ValueError: Если список чанков пуст
            RuntimeError: При ошибках построения индексов
            
        Процесс:
            1. Валидация входных данных
            2. Проверка кэша и загрузка при совпадении хэша
            3. Построение BM25 индекса
            4. Построение FAISS индекса
            5. Сохранение в кэш
        """
        if not chunks:
            raise ValueError("❌ Список чанков не может быть пустым")
        
        logger.info(f"🏗️ Построение индексов для {len(chunks)} чанков...")
        
        # Сохраняем корпус
        self.chunk_store = chunks
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Преобразуем чанки в формат LangChain
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "id": chunk.chunk_id,
                    "path": chunk.source_path
                }
            )
            for chunk in chunks
        ]
        
        # Попытка загрузки из кэша
        if use_cache and self._try_load_from_cache():
            return
        
        try:
            # Построение BM25 индекса
            logger.info("📝 Построение BM25 индекса...")
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = self.basic_search_k
            
            # Построение FAISS индекса
            logger.info("🧠 Построение FAISS индекса...")
            self.faiss_store = FAISS.from_documents(documents, self.embeddings)
            
            # Сохранение в кэш
            if use_cache:
                self._save_to_cache()
                
            logger.info("✅ Все индексы успешно построены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка построения индексов: {e}")
            raise RuntimeError(f"Не удалось построить индексы: {e}")

    def _try_load_from_cache(self) -> bool:
        """
        Пытается загрузить индексы из кэша.
        
        Returns:
            bool: True если загрузка успешна, False иначе
        """
        cache_path = os.path.join(self.cache_dir, "faiss_index")
        hash_path = os.path.join(self.cache_dir, "hash.txt")
        
        # Проверяем существование файлов кэша
        if not (os.path.exists(cache_path) and os.path.exists(hash_path)):
            logger.debug("💾 Кэш не найден")
            return False
        
        try:
            # Проверяем совпадение хэша
            with open(hash_path, "r", encoding="utf-8") as f:
                cached_hash = f.read().strip()
                
            current_hash = self._get_corpus_hash()
            
            if cached_hash != current_hash:
                logger.info("🔄 Корпус изменился, пересоздание индексов...")
                return False
            
            # Загружаем FAISS индекс
            logger.info("💾 Загрузка FAISS индекса из кэша...")
            self.faiss_store = FAISS.load_local(
                cache_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Восстанавливаем BM25 (быстро пересоздаём)
            documents = [
                Document(
                    page_content=chunk.text,
                    metadata={"id": chunk.chunk_id, "path": chunk.source_path}
                )
                for chunk in self.chunk_store
            ]
            
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = self.basic_search_k
            
            logger.info("✅ Индексы успешно загружены из кэша")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки кэша: {e}")
            return False

    def _save_to_cache(self) -> None:
        """
        Сохраняет индексы в кэш для быстрого перезапуска.
        """
        try:
            # Создаём папку кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Сохраняем FAISS индекс
            cache_path = os.path.join(self.cache_dir, "faiss_index")
            self.faiss_store.save_local(cache_path)
            
            # Сохраняем хэш корпуса
            hash_path = os.path.join(self.cache_dir, "hash.txt")
            with open(hash_path, "w", encoding="utf-8") as f:
                f.write(self._get_corpus_hash())
            
            logger.info(f"💾 Индексы сохранены в кэш: {self.cache_dir}")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка сохранения кэша: {e}")

    def _documents_to_chunks(self, documents: List[Document]) -> List[Chunk]:
        """
        Преобразует LangChain документы обратно в объекты Chunk.
        
        Args:
            documents: Список LangChain документов
            
        Returns:
            List[Chunk]: Список объектов Chunk
        """
        chunks = []
        
        for doc in documents:
            chunk_id = doc.metadata.get('id')
            
            # Используем сохранённый чанк если доступен
            if chunk_id in self.chunk_by_id:
                chunks.append(self.chunk_by_id[chunk_id])
            else:
                # Создаём новый чанк из документа
                chunks.append(Chunk(
                    text=doc.page_content,
                    source_path=doc.metadata.get('path', 'unknown'),
                    chunk_id=chunk_id
                ))
        
        return chunks

    def search(self, query: str) -> List[Chunk]:
        """
        Выполняет гибридный поиск с переранжированием.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            List[Chunk]: Топ релевантных чанков
            
        Raises:
            RuntimeError: Если индексы не построены
            
        Алгоритм:
            1. Гибридный поиск: FAISS + BM25 с равными весами
            2. Объединение и дедупликация результатов
            3. Нейронное переранжирование топ кандидатов
            4. Возврат топ-k наиболее релевантных чанков
        """
        if not self.faiss_store or not self.bm25_retriever:
            raise RuntimeError("❌ Индексы не построены. Вызовите build() сначала.")
        
        logger.info(f"🔍 Поиск по запросу: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        try:
            # Этап 1: Гибридный поиск
            ensemble_retriever = EnsembleRetriever(
                retrievers=[
                    self.bm25_retriever,  # Лексический поиск
                    self.faiss_store.as_retriever(
                        search_kwargs={'k': self.basic_search_k}
                    )  # Семантический поиск
                ],
                weights=[0.5, 0.5]  # Равные веса для обоих методов
            )
            
            # Получаем кандидатов
            retrieved_documents = ensemble_retriever.invoke(query)
            retrieved_chunks = self._documents_to_chunks(retrieved_documents)
            
            logger.info(f"📚 Найдено кандидатов: {len(retrieved_chunks)}")
            
            # Этап 2: Переранжирование
            if self.reranker and retrieved_chunks:
                ranked_chunks = self.reranker.rerank_documents(
                    query, 
                    retrieved_chunks, 
                    num_docs=self.top_k
                )
                
                logger.info(f"🎯 Финальный результат: {len(ranked_chunks)} чанков")
                
                return ranked_chunks
            else:
                # Без переранжирования
                logger.warning("⚠️ Reranker недоступен, возврат без переранжирования")
                return retrieved_chunks[:self.top_k]
        
        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            return []

    def get_search_stats(self) -> Dict[str, any]:
        """
        Возвращает статистику поискового движка.
        
        Returns:
            Dict: Словарь с метриками и настройками
        """
        return {
            "total_chunks": len(self.chunk_store),
            "top_k": self.top_k,
            "basic_search_k": self.basic_search_k,
            "faiss_ready": self.faiss_store is not None,
            "bm25_ready": self.bm25_retriever is not None,
            "reranker_ready": self.reranker is not None,
            "cache_dir": self.cache_dir
        }

# =============================================================================
# ФУНКЦИИ ДЛЯ ТЕСТИРОВАНИЯ
# =============================================================================

def test_retriever() -> None:
    """
    Простой тест retriever'а с примером данных.
    """
    logger.info("🧪 Тестирование retriever'а...")
    
    # Создаём тестовые чанки
    test_chunks = [
        Chunk(
            text="Херсон находится в зоне с весом снежного покрова 0,5 кПа/м².",
            source_path="/test/doc1.docx",
            chunk_id=1
        ),
        Chunk(
            text="Мелитополь имеет вес снежного покрова 0,95 кПа/м².",
            source_path="/test/doc2.docx", 
            chunk_id=2
        )
    ]
    
    # Тестируем retriever
    retriever = Retriever(top_k=2)
    retriever.build(test_chunks, use_cache=False)
    
    test_query = "вес снежного покрова Херсон"
    results = retriever.search(test_query)
    
    print(f"Запрос: {test_query}")
    print(f"Найдено результатов: {len(results)}")
    
    for i, chunk in enumerate(results, 1):
        score = getattr(chunk, 'score', 'N/A')
        print(f"{i}. Оценка: {score}, Текст: {chunk.text[:50]}...")


if __name__ == "__main__":
    # Запуск тестирования при прямом вызове модуля
    test_retriever()