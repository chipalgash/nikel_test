from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

from app.config import SETTINGS
from app.ingest import build_corpus
from app.questions import QUESTIONS
from app.retriever import HybridRetriever


logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_retrieval")


def _needs_table_boost(question: str) -> bool:
    q = (question or "").lower()
    markers = (
        "в каких зонах",
        "k_h",
        "коэффициент",
        "таблиц",
        "в каком пункте",
        "толщина защитного слоя",
    )
    return any(m in q for m in markers)


def run(question: str | None = None, rebuild: bool = False, top_n: int = 8) -> str:
    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Retrieval-only run started | rebuild=%s | top_n=%s | data_dir=%s",
        rebuild,
        top_n,
        SETTINGS.data_dir,
    )

    chunks = build_corpus(SETTINGS.data_dir)
    retriever = HybridRetriever()
    retriever.build(chunks, rebuild=rebuild)

    questions = [question] if question else QUESTIONS
    rows = []
    for i, q in enumerate(questions, start=1):
        logger.info("Question %s/%s | text=%s", i, len(questions), q)
        source_hints = retriever.suggested_source_hints(q)
        hits = retriever.search(q, source_hints=source_hints)
        if _needs_table_boost(q):
            table_hits = retriever.search_table(q, top_k=max(top_n * 2, 12), source_hints=source_hints)
            if table_hits:
                hits = retriever.merge_chunk_lists(table_hits, hits)
        hits = hits[:top_n]

        rows.append(
            {
                "question": q,
                "source_hints": source_hints,
                "retrieved_count": len(hits),
                "retrieved": [
                    {
                        "chunk_id": c.chunk_id,
                        "chunk_role": c.chunk_role,
                        "parent_id": c.parent_id,
                        "source_file": c.source_file,
                        "section_path": c.section_path,
                        "block_types": c.block_types,
                        "text": c.text,
                    }
                    for c in hits
                ],
            }
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SETTINGS.results_dir / f"retrieval_{ts}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Retrieval-only run finished | file=%s", out_path)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-only run without generation")
    parser.add_argument("--question", type=str, default=None, help="Single custom question")
    parser.add_argument("--rebuild", action="store_true", help="Force reindex")
    parser.add_argument("--top-n", type=int, default=8, help="How many raw chunks to save per question")
    args = parser.parse_args()

    run(question=args.question, rebuild=args.rebuild, top_n=args.top_n)


if __name__ == "__main__":
    main()
