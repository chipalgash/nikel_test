from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    cache_dir: Path = project_root / "cache"
    results_dir: Path = project_root / "results"

    embed_model: str = os.getenv("EMBED_MODEL", "bge-m3")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    gen_model: str = os.getenv("GEN_MODEL", "qwen2.5:14b-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    legacy_strategy: bool = os.getenv("LEGACY_STRATEGY", "0") == "1"

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900" if os.getenv("LEGACY_STRATEGY", "0") == "1" else "1800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200" if os.getenv("LEGACY_STRATEGY", "0") == "1" else "180"))
    hybrid_k: int = int(os.getenv("HYBRID_K", "20" if os.getenv("LEGACY_STRATEGY", "0") == "1" else "30"))
    final_k: int = int(os.getenv("FINAL_K", "10" if os.getenv("LEGACY_STRATEGY", "0") == "1" else "8"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.5" if os.getenv("LEGACY_STRATEGY", "0") == "1" else "0.6"))
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "0.5" if os.getenv("LEGACY_STRATEGY", "0") == "1" else "0.4"))
    adaptive_hybrid_weights: bool = os.getenv("ADAPTIVE_HYBRID_WEIGHTS", "0") == "1"
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "24"))
    use_hf_embeddings: bool = os.getenv("USE_HF_EMBEDDINGS", "0") == "1"
    enable_reranker: bool = os.getenv("ENABLE_RERANKER", "1") == "1"
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()


SETTINGS = Settings()
