from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from urllib3.util.request import ChunksAndContentLength
from models import Chunk
from pydantic import BaseModel
import logging
from typing import List



logger = logging.getLogger(__name__)

class LLMResponse(BaseModel):
    answer: str
    chunks_id: list[int]


class ValidationScore(BaseModel):
    index: int
    score: float


class ValidationResults(BaseModel):
    results: List[ValidationScore]


class Generator:
    def __init__(self) -> None:
        self.llm = ChatOllama(
            model='qwen3:14b',
            reasoning=False,
            temperature=0,
            num_predict=1024,
            repeat_penalty=1.1,
            format=LLMResponse.model_json_schema()
        )

    def get_answer_sources(self, chunks: list[Chunk], relevant_ids: list[str]) -> str:
        sources = []
        seen_sources = set()
        
        logger.info(f"Поиск источников для {len(chunks)} чанков, релевантных ID: {relevant_ids}")
        
        for chunk in chunks:
            if chunk.chunk_id in relevant_ids:
                if chunk.source_path not in seen_sources:
                    sources.append(chunk.source_path)
                    seen_sources.add(chunk.source_path)
        
        result = 'Источники:\n\n' + '\n'.join([str(i) + '.' + ' ' + path for i, path in enumerate(sources, start=1)])
        
        return result


    def generate_answer(self, query: str, chunks: list[Chunk]) -> tuple[str, list[int]]:
        if not chunks:
            return ('Отсутствует информация в предоставленном контексте.', [])

        context = '\n\n'.join(
            ['ID чанка: ' + str(chunk.chunk_id) + '\n' + chunk.text + '\n\n' for chunk in chunks] 
        )
        messages = [
                SystemMessage(
                content="""Ты — ассистент, отвечающий на вопросы на русском языке.

ПРАВИЛА:
• Отвечай ТОЛЬКО на основе предоставленного контекста
• Отвечай кратко (1-2 предложения)
• Используй ТОЛЬКО русский язык
• Приводи номера пунктов и названия таблиц
• Если информации нет - скажи "Недостаточно информации"

ФОРМАТ: Краткий фактологический ответ без лишних слов."""
                ), 
                HumanMessage(
                    content=f"Контекст:\n{context}\n\nВопрос: {query}\n\nОтветь кратко на русском языке, используя только контекст. Верни JSON:\n{{'answer': 'краткий ответ', 'chunks_id': [номера чанков, которые содержат информацию для ответа]}}"
                )
        ]

        try:
            response = self.llm.invoke(messages)
            logger.info(f'Ответ: {response.content}')
            response_data = LLMResponse.model_validate_json(response.content)
            
            # Логируем информацию о чанках
            logger.info(f"Получены чанки с ID: {response_data.chunks_id}")
            
            sources_str = self.get_answer_sources(chunks, response_data.chunks_id)
            ans = response_data.answer + '\n' + sources_str
            srcs = [
                {
                    'chunk_id': int(chunk.chunk_id),  # Преобразуем chunk_id в int
                    'text': chunk.text,
                    'file': chunk.source_path[chunk.source_path.rindex('/') + 1:]
                }
                for chunk in chunks
            ]
            return (ans, srcs)

        except Exception as e:
            logger.error(f'Ошибка во время генерации ответа {e}')

    def validate_answers_batch(self, questions: List[str], generated_answers: List[str], reference_answers: List[str]) -> List[ValidationScore]:
        """Валидация всех ответов одним запросом для ускорения процесса."""
        
        validation_prompt = (
            "Ты — эксперт по проверке фактологической точности ответов на технические вопросы.\n\n"
            "Задача: оценить, насколько сгенерированный ответ соответствует эталонному по фактам (не по стилю/формату).\n\n"
            "Оценка по шкале от 0 до 1:\n"
            "- 0.0-0.3: ответ неверный или содержит грубые ошибки\n"
            "- 0.4-0.6: ответ частично верный, но с неточностями\n"
            "- 0.7-0.9: ответ в целом верный, небольшие неточности\n"
            "- 1.0: ответ полностью соответствует эталону по фактам\n\n"
            "Верни JSON-массив с оценками для каждого ответа:\n"
            "[{\"index\": 0, \"score\": 0.8}, {\"index\": 1, \"score\": 0.9}]\n\n"
            "Вопросы и ответы:\n"
        )
        
        for i, (q, gen_ans, ref_ans) in enumerate(zip(questions, generated_answers, reference_answers)):
            validation_prompt += f"\n{i}. Вопрос: {q}\nЭталон: {ref_ans}\nСгенерировано: {gen_ans}\n"
        
        messages = [
                    SystemMessage(
                    content="Ты - эксперт по валидации. Отвечай только валидным JSON"
                    ), 
                    HumanMessage(content=validation_prompt)
            ]
        
        try:
            resp = self.llm.invoke(messages)
            logger.info(f"Ответ валидации: {resp.content}")
            
            # Пытаемся парсить ValidationResults
            validation_results = ValidationResults.model_validate_json(resp.content)
            return validation_results.results
            
        except Exception as e:
            logger.error(f"Ошибка валидации: {str(e)}")
            logger.error(f"Содержимое ответа: {resp.content if 'resp' in locals() else 'Нет ответа'}")
            
            # Возвращаем дефолтные значения
            return [ValidationScore(index=i, score=0.0) for i in range(len(questions))]


    


        