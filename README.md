# RAG Prototype

Open-source RAG pipeline for questions from `тз.md` on documents in `data/`.

## Pipeline
- Ingestion from DOCX paragraphs and tables.
- Structural chunking with section-aware metadata.
- Hybrid retrieval: BM25 + FAISS (bge-m3 via Ollama embeddings).
- Reranking: `BAAI/bge-reranker-v2-m3`.
- Generation: `qwen2.5:14b-instruct` via Ollama with JSON output.
- Output with short answer and explicit source links (`file | section | chunk_id`).

Legacy-like profile (close to candidate strategy):
- `LEGACY_STRATEGY=1` switches defaults to `CHUNK_SIZE=900`, `CHUNK_OVERLAP=200`, `HYBRID_K=20`, `FINAL_K=10`, `BM25_WEIGHT=0.5`, `DENSE_WEIGHT=0.5`.
- Hybrid merge uses LangChain `EnsembleRetriever` with fixed BM25/FAISS weights.
- Optional query-adaptive weights are off by default (`ADAPTIVE_HYBRID_WEIGHTS=0`).

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# full run on 10 questions
python run_rag.py --rebuild

# single question
python run_rag.py --question "Сколько разделов должна содержать проектная документация согласно 87-му постановлению?"
```

Results are saved to `results/run_YYYYMMDD_HHMMSS.json`.
