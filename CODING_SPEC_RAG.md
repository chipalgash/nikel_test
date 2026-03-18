
# CODING_SPEC.md
## RAG система для ответов на вопросы по нормативным документам

---

# 1. Контекст проекта

Проект представляет собой прототип Retrieval-Augmented Generation (RAG) системы, предназначенной для ответа на вопросы по нормативно‑техническим документам из области строительства и геодезии.

Система должна анализировать документы, извлекать релевантные фрагменты и генерировать краткий ответ на вопрос с указанием источника.

Основные технологии проекта:

- Python
- LangChain
- Langfuse
- Yandex Cloud API (LLM + Embeddings)
- Vector Index (Yandex или FAISS fallback)

---

## Структура проекта

project/

src/
    ingestion/
    chunking/
    embeddings/
    retriever/
    rag/
    evaluation/

data/
    docs/
    questions/
    gold/

artifacts/
    index/
    chunks/

results/

tests/

.env
README.md

---

# 2. Текущая реализация и проблема

На данный момент имеется пример реализации RAG системы другим разработчиком.

Особенности этой реализации:

- DOCX ingestion
- table-aware chunking
- retrieval
- reranking
- генерация ответа

Однако данное решение использует:

- rule-based routing
- source hints
- hardcoded правила для конкретных вопросов
- специальные query injections

Такая архитектура привязана к конкретному набору вопросов.

Задача данного проекта — реализовать систему, которая:

- работает без эвристик
- масштабируется на новые вопросы
- использует чистый retrieval + reasoning

---

# 3. Цель задачи

Разработать MVP прототип RAG системы, который:

1. индексирует нормативные DOCX документы
2. извлекает релевантные фрагменты
3. генерирует краткий ответ на вопрос
4. указывает источник ответа

Система должна корректно отвечать на 10 тестовых вопросов, но не должна быть жёстко привязана к ним.

Тип задачи: создание новой функциональности.

---

# 4. Требования к функционалу

Pipeline системы:

DOCX documents  
↓  
Document loader  
↓  
DOCX parser  
↓  
Chunking + metadata  
↓  
Embeddings  
↓  
Vector index  
↓  
Retriever  
↓  
LLM (RAG)  
↓  
Answer + sources

---

## 4.1 Document ingestion

Система должна читать `.docx` документы из папки:

data/docs/

Необходимо извлекать:

- текст
- заголовки
- таблицы
- порядок элементов документа

---

## 4.2 Chunking

Реализовать table-aware chunking.

Типы чанков:

- text
- table_row
- table_context

Пример metadata:

{
  "document": "SP_20.13330.2016.docx",
  "chunk_id": "doc_12_chunk_45",
  "section": "Раздел 4",
  "subsection": "4.1",
  "block_type": "text",
  "text": "..."
}

Параметры:

chunk_size = 800  
chunk_overlap = 100

---

# 5. Retrieval

Используется dense retrieval.

Pipeline:

question → embedding → vector search → top_k chunks

Параметры:

top_k = 5

Vector index:

- Yandex Vector Search
или
- FAISS fallback

---

# 6. Генерация ответа

LLM получает:

- question
- retrieved_chunks

Модель должна:

1. Найти ответ строго в retrieved context
2. Не использовать внешние знания

Если ответ отсутствует — вернуть:

"ответ не найден"

---

# 7. Формат ответа

Ответ возвращается в JSON.

Пример:

{
  "question_id": 3,
  "question": "Выведи рекомендуемые варианты конструктивного решения заземлителей",
  "answer": "Рекомендуемые варианты: ...",
  "sources": [
    {
      "document": "СП_153.13130.2013.docx",
      "chunk_id": "chunk_87",
      "section": "6.2",
      "justification": "Ответ извлечен из фрагмента документа."
    }
  ]
}

Если ответа нет:

{
  "question_id": 5,
  "answer": "ответ не найден",
  "sources": []
}

---

# 8. CLI интерфейс

Один вопрос:

python main.py --question "..."

Batch режим:

python main.py --questions data/questions/questions.json

Результат:

results/answers.json

---

# 9. Logging

Используется Langfuse.

Логируются:

- ingestion
- chunking
- embeddings
- retrieval
- generation
- evaluation

---

# 10. Evaluation

Сравнение с:

data/gold/gold_answers.json

Проверяется:

- корректность ответа
- корректность источника
- корректность section/chunk

Выход:

results/evaluation_report.json

---

# 11. Ограничения

Запрещено:

- rule-based routing
- mapping вопрос → документ
- regex extraction под конкретные вопросы
- hardcoded ответы
- keyword routing

Разрешено:

retrieval + LLM reasoning

---

# 12. Архитектура и код

Код должен:

- быть модульным
- иметь docstrings
- быть документированным
- легко расширяться

---

# 13. Конфигурация

.env пример:

YANDEX_API_KEY=
YANDEX_FOLDER_ID=

VECTOR_DB=faiss
TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=100

---

# 14. Артефакты

Сохраняются:

artifacts/parsed_documents.json  
artifacts/chunks.json  
artifacts/vector_index/  
artifacts/retrieval_results.json  

results/answers.json  
results/evaluation_report.json  

---

# 15. Критерии приемки

MVP считается готовым если:

1. документы индексируются
2. retrieval работает
3. система отвечает на 10 тестовых вопросов
4. указывает источники
5. работает CLI
6. результаты сохраняются
7. используется Langfuse
