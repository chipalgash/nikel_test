from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from app.config import SETTINGS
from app.generate import AnswerGenerator
from app.schemas import Chunk


logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_generate_from_retrieval")


def _load_retrieval(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Retrieval file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Retrieval file must contain a list")
    return data


def _row_to_chunks(row: dict) -> list[Chunk]:
    out: list[Chunk] = []
    for item in row.get("retrieved", []):
        out.append(
            Chunk(
                chunk_id=int(item["chunk_id"]),
                text=str(item["text"]),
                source_path=str(item.get("source_path", "")),
                source_file=str(item.get("source_file", "")),
                section_path=str(item.get("section_path", "")),
                block_types=list(item.get("block_types", [])),
                chunk_role=str(item.get("chunk_role", "parent")),
                parent_id=(int(item["parent_id"]) if item.get("parent_id") is not None else None),
            )
        )
    return out


def run(retrieval_file: Path, limit: int | None = None) -> Path:
    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generation from retrieval started | file=%s | limit=%s", retrieval_file, limit)

    rows = _load_retrieval(retrieval_file)
    if limit is not None:
        rows = rows[:limit]

    generator = AnswerGenerator()
    results = []
    for i, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        chunks = _row_to_chunks(row)
        logger.info("Question %s/%s | chunks=%s | text=%s", i, len(rows), len(chunks), question)
        answer = generator.answer(question, chunks)
        result = answer.to_dict()
        result["source_hints"] = row.get("source_hints", [])
        results.append(result)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SETTINGS.results_dir / f"run_from_retrieval_{ts}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Generation from retrieval finished | file=%s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers from previously saved retrieval results")
    parser.add_argument("--retrieval-file", type=str, required=True, help="Path to retrieval_*.json")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N questions")
    args = parser.parse_args()

    run(Path(args.retrieval_file), limit=args.limit)


if __name__ == "__main__":
    main()
