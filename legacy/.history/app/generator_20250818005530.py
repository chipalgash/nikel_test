#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор ответов на основе LLM
===============================

Модуль для генерации ответов на вопросы с использованием больших языковых моделей
и автоматической валидации качества ответов.

Основные компоненты:
    - Generator: Класс для генерации ответов с извлечением источников
    - Валидация: Автоматическая оценка качества ответов
    - Структурированный вывод: JSON формат с метаданными

Автор: AI Assistant  
Дата: 2024
"""

import logging
from typing import List, Tuple, Dict, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from models import Chunk

# Настройка логирования
logger = logging.getLogger(__name__)

# =============================================================================
# МОДЕЛИ ДАННЫХ ДЛЯ СТРУКТУРИРОВАННОГО ВЫВОДА
# =============================================================================

class LLMResponse(BaseModel):
    """
    Структура ответа LLM с указанием использованных источников.
    
    Attributes:
        answer: Текст ответа на вопрос
        chunks_id: Список ID чанков, использованных для ответа
    """
    answer: str
    chunks_id: List[int]


class ValidationScore(BaseModel):
    """
    Оценка качества одного ответа.
    
    Attributes:
        question_index: Индекс вопроса в списке
        score: Оценка от 0.0 до 1.0
    """
    question_index: int
    score: float


class ValidationResults(BaseModel):
    """
    Результаты валидации для всех ответов.
    
    Attributes:
        results: Список оценок для каждого вопроса
    """
    results: List[ValidationScore]

# =============================================================================
# ОСНОВНОЙ КЛАСС ГЕНЕРАТОРА
# =============================================================================

class Generator:
    """
    Генератор ответов с использованием LLM и автоматической валидацией.
    
    Возможности:
        - Генерация кратких фактологических ответов
        - Автоматическое извлечение источников
        - Структурированный JSON вывод
        - Валидация качества ответов
    
    Модель: Qwen3:14b через Ollama (open-source)
    """
    
    def __init__(self) -> None:
        """
        Инициализация генератора с настройками модели.
        
        Настройки модели:
            - temperature=0: Детерминированные ответы
            - num_predict=1024: Максимальная длина ответа
            - repeat_penalty=1.1: Избежание повторений
        """
        logger.info("🤖 Инициализация LLM генератора...")
        
        try:
            self.llm = ChatOllama(
                model='qwen3:14b',
                reasoning=False,          # Отключаем chain-of-thought для скорости
                temperature=0,            # Детерминированность для технических ответов
                num_predict=1024,         # Достаточно для кратких ответов
                repeat_penalty=1.1,       # Избегаем повторений
                format=LLMResponse.model_json_schema()  # Структурированный вывод
            )
            
            logger.info("✅ LLM генератор готов к работе")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации LLM: {e}")
            raise

    def get_answer_sources(self, chunks: List[Chunk], relevant_ids: List[int]) -> str:
        """
        Извлекает уникальные источники документов для ответа.
        
        Args:
            chunks: Список всех чанков, переданных в LLM
            relevant_ids: ID чанков, которые LLM указал как релевантные
            
        Returns:
            str: Форматированный список источников
            
        Логика:
            1. Находит чанки с указанными ID
            2. Извлекает уникальные пути к документам
            3. Форматирует в виде нумерованного списка
        """
        logger.debug(f"🔍 Поиск источников для {len(chunks)} чанков, релевантных ID: {relevant_ids}")
        
        sources = []
        seen_sources = set()
        
        # Ищем чанки с релевантными ID
        for chunk in chunks:
            if chunk.chunk_id in relevant_ids:
                source_path = chunk.source_path
                
                # Добавляем только уникальные источники
                if source_path not in seen_sources:
                    sources.append(source_path)
                    seen_sources.add(source_path)
        
        # Форматируем список источников
        if sources:
            sources_text = 'Источники:\n\n' + '\n'.join([
                f"{i}. {source}" 
                for i, source in enumerate(sources, start=1)
            ])
        else:
            sources_text = 'Источники: Не указаны'
            
        logger.debug(f"📚 Найдено уникальных источников: {len(sources)}")
        return sources_text

    def generate_answer(self, query: str, chunks: List[Chunk]) -> Tuple[str, List[Dict]]:
        """
        Генерирует ответ на вопрос на основе релевантных чанков.
        
        Args:
            query: Текст вопроса
            chunks: Список релевантных чанков документов
            
        Returns:
            Tuple[str, List[Dict]]: Ответ с источниками и метаданные чанков
            
        Raises:
            Exception: При ошибках взаимодействия с LLM
            
        Процесс:
            1. Формирование контекста из чанков
            2. Создание промпта с правилами
            3. Вызов LLM с принудительным JSON форматом
            4. Извлечение источников
            5. Формирование итогового ответа
        """
        logger.info(f"💭 Генерация ответа для вопроса: {query[:50]}...")
        
        # Проверяем наличие чанков
        if not chunks:
            logger.warning("⚠️ Нет релевантных чанков для ответа")
            return ('Отсутствует информация в предоставленном контексте.', [])

        # Формируем контекст из всех чанков
        context_parts = []
        for chunk in chunks:
            context_parts.append(f'ID чанка: {chunk.chunk_id}\n{chunk.text}\n')
        
        context = '\n'.join(context_parts)
        
        # Создаём промпт с чёткими правилами
        system_prompt = """Ты — ассистент, отвечающий на вопросы на русском языке.

ПРАВИЛА:
• Отвечай ТОЛЬКО на основе предоставленного контекста
• Отвечай кратко (1-2 предложения)
• Используй ТОЛЬКО русский язык
• Приводи номера пунктов и названия таблиц
• Если информации нет - скажи "Недостаточно информации"

ФОРМАТ: Краткий фактологический ответ без лишних слов."""

        user_prompt = f"""Контекст:
{context}

Вопрос: {query}

Ответь кратко на русском языке, используя только контекст. 
Верни JSON:
{{'answer': 'краткий ответ', 'chunks_id': [номера чанков, которые содержат информацию для ответа]}}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            # Вызываем LLM с принудительным JSON форматом
            logger.debug("🔄 Отправка запроса к LLM...")
            response = self.llm.invoke(messages)
            
            logger.debug(f"📝 Получен ответ: {response.content[:100]}...")
            
            # Парсим структурированный ответ
            response_data = LLMResponse.model_validate_json(response.content)
            
            logger.info(f"🎯 Использованы чанки с ID: {response_data.chunks_id}")
            
            # Извлекаем источники
            sources_text = self.get_answer_sources(chunks, response_data.chunks_id)
            
            # Формируем итоговый ответ
            final_answer = response_data.answer + '\n' + sources_text
            
            # Создаём метаданные для всех чанков
            sources_metadata = [
                {
                    'chunk_id': int(chunk.chunk_id),
                    'text': chunk.text,
                    'file': chunk.source_path.split('/')[-1]  # Только имя файла
                }
                for chunk in chunks
            ]
            
            logger.info("✅ Ответ успешно сгенерирован")
            return (final_answer, sources_metadata)

        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            error_response = f"Ошибка генерации ответа: {str(e)}"
            return (error_response, [])

    def validate_answers_batch(
        self, 
        questions: List[str], 
        generated_answers: List[str], 
        reference_answers: List[str]
    ) -> List[ValidationScore]:
        """
        Валидация качества всех ответов одним запросом к LLM.
        
        Args:
            questions: Список исходных вопросов
            generated_answers: Список сгенерированных ответов
            reference_answers: Список эталонных ответов
            
        Returns:
            List[ValidationScore]: Оценки качества для каждого ответа
            
        Критерии оценки:
            - 0.0-0.3: Неверный или содержит грубые ошибки
            - 0.4-0.6: Частично верный, но с неточностями  
            - 0.7-0.9: В целом верный, небольшие неточности
            - 1.0: Полностью соответствует эталону
        """
        logger.info(f"🔍 Валидация {len(questions)} ответов...")
        
        # Формируем подробный промпт для валидации
        validation_prompt = (
            "Ты — эксперт по проверке фактологической точности ответов на технические вопросы.\n\n"
            "Задача: оценить, насколько сгенерированный ответ соответствует эталонному по фактам "
            "(не по стилю/формату).\n\n"
            "Оценка по шкале от 0 до 1:\n"
            "- 0.0-0.3: ответ неверный или содержит грубые ошибки\n"
            "- 0.4-0.6: ответ частично верный, но с неточностями\n"
            "- 0.7-0.9: ответ в целом верный, небольшие неточности\n"
            "- 1.0: ответ полностью соответствует эталону по фактам\n\n"
            "Верни JSON-массив с оценками для каждого ответа:\n"
            '{"results": [{"question_index": 0, "score": 0.8}, {"question_index": 1, "score": 0.9}]}\n\n'
            "Вопросы и ответы:\n"
        )
        
        # Добавляем все пары вопрос-ответ
        for i, (question, generated, reference) in enumerate(zip(questions, generated_answers, reference_answers)):
            validation_prompt += (
                f"\n{i}. Вопрос: {question}\n"
                f"Эталон: {reference}\n"
                f"Сгенерировано: {generated}\n"
            )
        
        messages = [
            SystemMessage(
                content="Ты - эксперт по валидации. Отвечай ТОЛЬКО валидным JSON в указанном формате."
            ),
            HumanMessage(content=validation_prompt)
        ]
        
        try:
            logger.debug("🔄 Отправка запроса валидации к LLM...")
            
            # Вызываем LLM для валидации
            response = self.llm.invoke(messages)
            
            # Парсим результаты валидации
            validation_results = ValidationResults.model_validate_json(response.content)
            
            logger.info(f"✅ Валидация завершена для {len(validation_results.results)} ответов")
            
            # Логируем средную оценку
            if validation_results.results:
                avg_score = sum(r.score for r in validation_results.results) / len(validation_results.results)
                logger.info(f"📊 Средняя оценка качества: {avg_score:.2f}")
            
            return validation_results.results
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации: {e}")
            
            # Возвращаем нулевые оценки при ошибке
            fallback_results = [
                ValidationScore(question_index=i, score=0.0) 
                for i in range(len(questions))
            ]
            
            logger.warning(f"⚠️ Возвращены нулевые оценки для {len(fallback_results)} вопросов")
            return fallback_results

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def test_generator() -> None:
    """
    Функция для тестирования генератора с примером.
    
    Использует тестовый вопрос и чанк для проверки работоспособности.
    """
    logger.info("🧪 Тестирование генератора...")
    
    # Создаём тестовые данные
    test_chunk = Chunk(
        text="Херсон находится в зоне с весом снежного покрова 0,5 кПа/м².",
        source_path="/test/document.docx",
        chunk_id=1
    )
    
    test_question = "Какой вес снежного покрова в Херсоне?"
    
    # Тестируем генератор
    generator = Generator()
    answer, sources = generator.generate_answer(test_question, [test_chunk])
    
    print(f"Вопрос: {test_question}")
    print(f"Ответ: {answer}")
    print(f"Источники: {len(sources)}")


if __name__ == "__main__":
    # Запуск тестирования при прямом вызове модуля
    test_generator()