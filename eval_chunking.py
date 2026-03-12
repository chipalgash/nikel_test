from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from app.config import SETTINGS
from app.ingest import build_corpus
from app.questions import QUESTIONS
from app.retriever import HybridRetriever


logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eval_chunking")


@dataclass
class QueryExpectation:
    groups: List[List[str]]
    source_hints: List[str]


EXPECTATIONS: List[QueryExpectation] = [
    QueryExpectation(groups=[["0.5", "0,5"], ["0.95", "0,95"]], source_hints=["сп 20.13330"]),
    QueryExpectation(groups=[["k_h", "kh", "коэффициент"], ["2"], ["адлер"]], source_hints=["сп 20.13330"]),
    QueryExpectation(groups=[["3.2", "3,2"], ["3.3", "3,3"]], source_hints=["всп 22-02-07"]),
    QueryExpectation(groups=[["техническ"], ["средств"]], source_hints=["всп 22-02-07"]),
    QueryExpectation(groups=[["технические решения"], ["схем"]], source_hints=["постановление_правительства"]),
    QueryExpectation(groups=[["13"]], source_hints=["постановление_правительства", "_87_"]),
    QueryExpectation(groups=[["20"], ["км/ч", "км ч", "кмч"]], source_hints=["приказ-ростехнадзора"]),
    QueryExpectation(groups=[["26"], ["c", "°c", "с"]], source_hints=["приказ-ростехнадзора"]),
    QueryExpectation(groups=[["таблиц", "табл"], ["4.1", "4,1"]], source_hints=["сп 69.13330", "сп 91.13330"]),
    QueryExpectation(groups=[["6.1.10", "6,1,10"], ["30"], ["мм"]], source_hints=["приказ-ростехнадзора"]),
]


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


def _normalize(text: str) -> str:
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"\s+", " ", t)
    return t


def _group_hit(group: List[str], haystack: str) -> bool:
    return any(_normalize(opt) in haystack for opt in group)


def _source_hit(source_hints: List[str], source_names: List[str]) -> bool:
    if not source_hints:
        return True
    joined = " | ".join(_normalize(x) for x in source_names)
    return any(_normalize(h) in joined for h in source_hints)


def run_eval(rebuild: bool = False, no_rerank: bool = True) -> Path:
    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Eval started | rebuild=%s", rebuild)

    chunks = build_corpus(SETTINGS.data_dir)
    retriever = HybridRetriever()
    retriever.build(chunks, rebuild=rebuild)
    if no_rerank:
        retriever.reranker = None
        logger.info("Reranker disabled for chunking eval")

    rows = []
    marker_hits = 0
    source_hits = 0
    both_hits = 0

    for i, question in enumerate(QUESTIONS, start=1):
        exp = EXPECTATIONS[i - 1]
        source_hints = retriever.suggested_source_hints(question)
        hits = retriever.search(question, source_hints=source_hints)
        if _needs_table_boost(question):
            table_hits = retriever.search_table(
                question,
                top_k=max(SETTINGS.final_k * 2, 12),
                source_hints=source_hints,
            )
            if table_hits:
                hits = retriever.merge_chunk_lists(table_hits, hits)[: max(SETTINGS.final_k * 2, SETTINGS.final_k)]

        context = _normalize("\n".join(h.text for h in hits))
        group_results = [bool(_group_hit(group, context)) for group in exp.groups]
        marker_ok = all(group_results)
        source_ok = _source_hit(exp.source_hints, [h.source_file for h in hits])
        both_ok = marker_ok and source_ok

        marker_hits += int(marker_ok)
        source_hits += int(source_ok)
        both_hits += int(both_ok)

        row = {
            "idx": i,
            "question": question,
            "marker_hit": marker_ok,
            "source_hit": source_ok,
            "combined_hit": both_ok,
            "group_results": group_results,
            "retrieved_chunk_ids": [h.chunk_id for h in hits],
            "retrieved_sources": sorted({h.source_file for h in hits}),
        }
        rows.append(row)
        logger.info(
            "Q%s | marker_hit=%s | source_hit=%s | combined=%s | sources=%s",
            i,
            marker_ok,
            source_ok,
            both_ok,
            row["retrieved_sources"][:3],
        )

    summary = {
        "questions_total": len(QUESTIONS),
        "marker_hit_at_k": marker_hits,
        "source_hit_at_k": source_hits,
        "combined_hit_at_k": both_hits,
    }
    logger.info("Summary | %s", summary)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SETTINGS.results_dir / f"chunk_eval_{ts}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Eval finished | file=%s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chunking/retrieval quality without generation")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild corpus/index before evaluation")
    parser.add_argument("--with-rerank", action="store_true", help="Enable reranker (slow); default is no reranker")
    args = parser.parse_args()
    run_eval(rebuild=args.rebuild, no_rerank=(not args.with_rerank))


if __name__ == "__main__":
    main()
