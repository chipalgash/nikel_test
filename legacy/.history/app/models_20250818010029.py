#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модели данных для RAG системы
=============================

Содержит классы данных для представления текстовых фрагментов
и связанных с ними метаданных в системе поиска и генерации ответов.

Основные модели:
    - Chunk: Базовая единица текстовой информации
    - ChunkWithScore: Чанк с оценкой релевантности
    - DocumentMetadata: Метаданные исходного документа

Автор: AI Assistant
Дата: 2024
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

# =============================================================================
# ОСНОВНЫЕ МОДЕЛИ ДАННЫХ
# =============================================================================

@dataclass
class Chunk:
    """
    Базовая единица текстовой информации в RAG системе.
    
    Представляет фрагмент документа с уникальным идентификатором
    и информацией об источнике.
    
    Attributes:
        text: Текстовое содержимое чанка
        source_path: Полный путь к исходному документу
        chunk_id: Уникальный числовой идентификатор чанка
        score: Оценка релевантности (добавляется reranker'ом)
        metadata: Дополнительные метаданные
    
    Примеры:
        >>> chunk = Chunk(
        ...     text="Херсон находится в зоне 0,5 по весу снежного покрова",
        ...     source_path="/data/regulations.docx",
        ...     chunk_id=42
        ... )
        >>> print(chunk.get_filename())
        'regulations.docx'
    """
    
    text: str
    source_path: str  
    chunk_id: int
    score: Optional[float] = field(default=None, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Валидация и нормализация данных после инициализации.
        
        Raises:
            ValueError: При некорректных входных данных
        """
        # Валидация обязательных полей
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Текст чанка не может быть пустым")
            
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("Путь к источнику не может быть пустым")
            
        if not isinstance(self.chunk_id, int) or self.chunk_id < 0:
            raise ValueError("ID чанка должен быть неотрицательным целым числом")
        
        # Нормализация текста
        self.text = self.text.strip()
        
        # Нормализация пути
        self.source_path = str(Path(self.source_path).resolve())
    
    def get_filename(self) -> str:
        """
        Извлекает имя файла из полного пути.
        
        Returns:
            str: Имя файла без пути
            
        Примеры:
            >>> chunk = Chunk("text", "/path/to/document.docx", 1)
            >>> chunk.get_filename()
            'document.docx'
        """
        return Path(self.source_path).name
    
    def get_file_extension(self) -> str:
        """
        Извлекает расширение файла.
        
        Returns:
            str: Расширение файла (например, '.docx')
        """
        return Path(self.source_path).suffix
    
    def is_valid(self) -> bool:
        """
        Проверяет валидность чанка.
        
        Returns:
            bool: True если чанк валиден
        """
        try:
            return (
                bool(self.text and self.text.strip()) and
                bool(self.source_path) and
                isinstance(self.chunk_id, int) and
                self.chunk_id >= 0
            )
        except (AttributeError, TypeError):
            return False
    
    def get_summary(self, max_length: int = 100) -> str:
        """
        Возвращает краткое описание чанка для отладки.
        
        Args:
            max_length: Максимальная длина текста в резюме
            
        Returns:
            str: Краткое описание чанка
        """
        text_preview = self.text[:max_length]
        if len(self.text) > max_length:
            text_preview += "..."
            
        score_info = f", score={self.score:.3f}" if self.score is not None else ""
        
        return f"Chunk(id={self.chunk_id}, file='{self.get_filename()}'{score_info}, text='{text_preview}')"
    
    def __str__(self) -> str:
        """
        Строковое представление чанка.
        
        Returns:
            str: Человекочитаемое описание
        """
        return self.get_summary()
    
    def __repr__(self) -> str:
        """
        Представление чанка для разработчиков.
        
        Returns:
            str: Детальное описание объекта
        """
        return (
            f"Chunk(text='{self.text[:30]}...', "
            f"source_path='{self.source_path}', "
            f"chunk_id={self.chunk_id}, "
            f"score={self.score})"
        )

# =============================================================================
# РАСШИРЕННЫЕ МОДЕЛИ
# =============================================================================

@dataclass
class ChunkWithScore(Chunk):
    """
    Чанк с обязательной оценкой релевантности.
    
    Используется после этапа переранжирования, когда каждый чанк
    имеет числовую оценку соответствия запросу.
    
    Attributes:
        score: Оценка релевантности от 0.0 до 1.0 (обязательная)
    """
    
    score: float = 0.0
    
    def __post_init__(self) -> None:
        """
        Дополнительная валидация оценки релевантности.
        """
        super().__post_init__()
        
        if not isinstance(self.score, (int, float)):
            raise ValueError("Оценка релевантности должна быть числом")
            
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("Оценка релевантности должна быть от 0.0 до 1.0")


@dataclass
class DocumentMetadata:
    """
    Метаданные исходного документа.
    
    Содержит информацию о документе-источнике для группировки
    и анализа чанков по документам.
    
    Attributes:
        file_path: Путь к файлу
        file_size: Размер файла в байтах
        creation_date: Дата создания файла
        total_chunks: Общее количество чанков в документе
        avg_chunk_length: Средняя длина чанка в символах
    """
    
    file_path: str
    file_size: Optional[int] = None
    creation_date: Optional[str] = None
    total_chunks: int = 0
    avg_chunk_length: float = 0.0
    
    def get_filename(self) -> str:
        """Извлекает имя файла из пути."""
        return Path(self.file_path).name
    
    def get_file_extension(self) -> str:
        """Извлекает расширение файла."""
        return Path(self.file_path).suffix

# =============================================================================
# УТИЛИТЫ ДЛЯ РАБОТЫ С ЧАНКАМИ
# =============================================================================

def create_chunk_from_text(
    text: str, 
    source_path: str, 
    chunk_id: int,
    **metadata
) -> Chunk:
    """
    Фабричная функция для создания чанка с валидацией.
    
    Args:
        text: Текстовое содержимое
        source_path: Путь к исходному файлу
        chunk_id: Уникальный идентификатор
        **metadata: Дополнительные метаданные
        
    Returns:
        Chunk: Новый объект чанка
        
    Raises:
        ValueError: При некорректных входных данных
    """
    chunk = Chunk(
        text=text,
        source_path=source_path,
        chunk_id=chunk_id,
        metadata=metadata
    )
    
    if not chunk.is_valid():
        raise ValueError(f"Не удалось создать валидный чанк: {chunk}")
    
    return chunk


def filter_chunks_by_score(
    chunks: List[Chunk], 
    min_score: float = 0.0,
    max_score: float = 1.0
) -> List[Chunk]:
    """
    Фильтрует чанки по диапазону оценок релевантности.
    
    Args:
        chunks: Список чанков для фильтрации
        min_score: Минимальная оценка релевантности
        max_score: Максимальная оценка релевантности
        
    Returns:
        List[Chunk]: Отфильтрованные чанки
    """
    return [
        chunk for chunk in chunks
        if chunk.score is not None and min_score <= chunk.score <= max_score
    ]


def group_chunks_by_source(chunks: List[Chunk]) -> Dict[str, List[Chunk]]:
    """
    Группирует чанки по исходным документам.
    
    Args:
        chunks: Список чанков для группировки
        
    Returns:
        Dict[str, List[Chunk]]: Словарь {имя_файла: [чанки]}
    """
    groups = {}
    
    for chunk in chunks:
        filename = chunk.get_filename()
        if filename not in groups:
            groups[filename] = []
        groups[filename].append(chunk)
    
    return groups


def calculate_chunk_statistics(chunks: List[Chunk]) -> Dict[str, Any]:
    """
    Вычисляет статистики по списку чанков.
    
    Args:
        chunks: Список чанков для анализа
        
    Returns:
        Dict[str, Any]: Словарь со статистиками
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_text_length": 0,
            "total_text_length": 0,
            "unique_sources": 0,
            "chunks_with_score": 0,
            "avg_score": None
        }
    
    # Базовые метрики
    total_chunks = len(chunks)
    text_lengths = [len(chunk.text) for chunk in chunks]
    total_text_length = sum(text_lengths)
    avg_text_length = total_text_length / total_chunks
    
    # Уникальные источники
    unique_sources = len(set(chunk.source_path for chunk in chunks))
    
    # Статистики по оценкам
    chunks_with_score = [chunk for chunk in chunks if chunk.score is not None]
    avg_score = None
    if chunks_with_score:
        avg_score = sum(chunk.score for chunk in chunks_with_score) / len(chunks_with_score)
    
    return {
        "total_chunks": total_chunks,
        "avg_text_length": avg_text_length,
        "total_text_length": total_text_length,
        "unique_sources": unique_sources,
        "chunks_with_score": len(chunks_with_score),
        "avg_score": avg_score,
        "min_text_length": min(text_lengths),
        "max_text_length": max(text_lengths)
    }

# =============================================================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# =============================================================================

def example_usage():
    """
    Демонстрация использования моделей данных.
    """
    print("🧪 Демонстрация моделей данных RAG системы\n")
    
    # Создание базового чанка
    chunk1 = Chunk(
        text="Херсон находится в зоне с весом снежного покрова 0,5 кПа/м².",
        source_path="/data/regulations.docx",
        chunk_id=1
    )
    
    print(f"1. Базовый чанк: {chunk1}")
    print(f"   Файл: {chunk1.get_filename()}")
    print(f"   Валидный: {chunk1.is_valid()}\n")
    
    # Добавление оценки релевантности
    chunk1.score = 0.85
    print(f"2. Чанк с оценкой: {chunk1.get_summary()}\n")
    
    # Создание чанка с метаданными
    chunk2 = create_chunk_from_text(
        text="Мелитополь имеет вес снежного покрова 0,95 кПа/м².",
        source_path="/data/regulations.docx",
        chunk_id=2,
        section="Приложение К",
        table="К.1"
    )
    chunk2.score = 0.92
    
    print(f"3. Чанк с метаданными: {chunk2}")
    print(f"   Метаданные: {chunk2.metadata}\n")
    
    # Анализ статистик
    chunks = [chunk1, chunk2]
    stats = calculate_chunk_statistics(chunks)
    
    print("4. Статистики чанков:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n5. Группировка по источникам:")
    groups = group_chunks_by_source(chunks)
    for filename, chunk_list in groups.items():
        print(f"   {filename}: {len(chunk_list)} чанков")


if __name__ == "__main__":
    # Запуск демонстрации при прямом вызове модуля
    example_usage()