from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Chunk:
    text: str
    source_path: str
    chunk_id: int
