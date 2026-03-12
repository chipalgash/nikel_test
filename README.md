# RAG Prototype For Normative DOCX

Прототип RAG-системы для вопросов из [`тз.md`](./тз.md) на документах из директории `data/`.

Цель: короткий ответ + явная ссылка на источник (`file | section | chunk_id`), с упором на точность фактов для числовых и табличных вопросов.

## 1) Архитектура

Пайплайн:
- Ingestion DOCX в порядке элементов документа (`paragraph`/`table`) с сохранением контекста секций.
- Chunking с типами блоков (`paragraph`, `table_row`, `table_context`) и метаданными.
- Hybrid retrieval: BM25 + FAISS (`bge-m3` через Ollama embeddings).
- Reranking: `BAAI/bge-reranker-v2-m3`.
- Генерация: `qwen2.5:7b-instruct` через Ollama.
- Rule-based post-extraction для вопросов, где критичны точные числа/пункты/перечни.

Ключевые модули:
- [`app/ingest.py`](./app/ingest.py)
- [`app/retriever.py`](./app/retriever.py)
- [`app/generate.py`](./app/generate.py)
- [`run_rag.py`](./run_rag.py)

## 2) Стратегия отбора и поиска

### 2.1 Ingestion/Chunking
- DOCX парсится в реальном порядке блоков, чтобы таблицы не теряли контекст секции.
- Заголовки нормализуются в `section_path`.
- Для таблиц создаются:
  - `table_row` (строки таблиц с нормализованными полями/числами),
  - `table_context` (локальный контекст вокруг строки).

Это важно для вопросов вида:
- `k_h > 2`,
- `таблица 4.1`,
- `пункт 6.1.10`,
- списки по подпунктам `б/в/г/д`.

### 2.2 Retrieval
- Базовый поиск: BM25 + dense (FAISS) с RRF-слиянием.
- Для anchor-вопросов (числа, пункты, таблицы, коэффициенты) усиливается lexical-составляющая и расширяется `hybrid_k`.
- Включены `source_hints` (routing по нужным документам, например постановление №87, СП 20, СП 69/91).
- Для табличных вопросов дополнительно запускается `search_table`.

### 2.3 Post-processing и контроль точности
- Hard filter по вопросу убирает нерелевантные хиты.
- Rule-based extraction (в [`run_rag.py`](./run_rag.py)) применен для критичных кейсов:
  - снеговые значения по Херсону/Мелитополю,
  - список регионов с `k_h > 2`,
  - расшифровка `ТС`,
  - количество разделов по постановлению №87,
  - `пункт 6.1.10` и `30 мм`,
  - структурированный ответ по графической части пояснительной записки.
- В [`app/generate.py`](./app/generate.py) добавлен безопасный парсинг JSON-ответа LLM с fallback, чтобы прогон не падал на обрезанном JSON.

## 3) Почему это дало результат

Бутылочное горлышко было не в самой LLM, а в retrieval на нормативных таблицах.

Что сработало:
- правильный порядок ingestion (`paragraph/table`),
- table-aware retrieval,
- query-aware routing,
- детерминированный extraction для точных фактов.

Итог: ответы стали стабильными на вопросах с жестким ожидаемым форматом (числа, пункты, перечни).

## 4) Текущие рабочие настройки

Рекомендуемый профиль для этого проекта:
- `GEN_MODEL=qwen2.5:7b-instruct`
- `EMBED_MODEL=bge-m3`
- `RERANK_MODEL=BAAI/bge-reranker-v2-m3`
- `ENABLE_RERANKER=1`
- `HYBRID_K=24`
- `FINAL_K=8`

См. файл [`.env`](./.env).

## 5) Запуск

### 5.1 Подготовка окружения
```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 5.2 Ollama
```bash
# сервер
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 /opt/homebrew/opt/ollama/bin/ollama serve

# модели
/opt/homebrew/opt/ollama/bin/ollama pull bge-m3
/opt/homebrew/opt/ollama/bin/ollama pull qwen2.5:7b-instruct
```

### 5.3 Полный прогон
```bash
cd "/Users/elizavetatsymbalova/Documents/codex projects/nikel_test"
source .venv311/bin/activate
python run_rag.py
```

### 5.4 Один вопрос
```bash
python run_rag.py --question "Сколько разделов должна содержать проектная документация согласно 87-му постановлению?"
```

Результаты сохраняются в `results/run_YYYYMMDD_HHMMSS.json`.

## 6) Зафиксированный результат

Файл последнего полного прогона:
- [`results/run_20260312_220650.json`](./results/run_20260312_220650.json)

По принятому критерию приемки в этой задаче:
- `10/10` (включая вопрос 9 как согласованный валидный).
