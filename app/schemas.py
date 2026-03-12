from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source_path: str
    source_file: str
    section_path: str
    block_types: List[str]
    chunk_role: str = "parent"
    parent_id: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=int(raw["chunk_id"]),
            text=str(raw["text"]),
            source_path=str(raw["source_path"]),
            source_file=str(raw["source_file"]),
            section_path=str(raw.get("section_path", "")),
            block_types=list(raw.get("block_types", [])),
            chunk_role=str(raw.get("chunk_role", "parent")),
            parent_id=(int(raw["parent_id"]) if raw.get("parent_id") is not None else None),
        )


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    chunk_id: int
    source_file: str
    section_path: str
    snippet: str


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunk_ids: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [asdict(x) for x in self.citations],
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
        }
