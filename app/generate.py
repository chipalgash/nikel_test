from __future__ import annotations

import logging
import re
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .config import SETTINGS
from .schemas import AnswerResult, Citation, Chunk


logger = logging.getLogger(__name__)


class GenerationOutput(BaseModel):
    answer_short: str = Field(description="Короткий ответ в 1-2 предложениях")
    citation_chunk_ids: List[int] = Field(description="Список ID чанков, использованных в ответе")


class AnswerGenerator:
    def __init__(self) -> None:
        self.llm = None
        self.use_llm = False
        try:
            self.llm = ChatOllama(
                model=SETTINGS.gen_model,
                base_url=SETTINGS.ollama_base_url,
                temperature=0,
                num_predict=700,
                format=GenerationOutput.model_json_schema(),
            )
            _ = self.llm.invoke([HumanMessage(content="ping")]).content
            self.use_llm = True
            logger.info("Using Ollama generator model: %s", SETTINGS.gen_model)
        except Exception as exc:
            logger.warning("Ollama generator unavailable, fallback to extractive mode: %s", exc)

    def _context(self, chunks: List[Chunk]) -> str:
        parts = []
        for c in chunks:
            parts.append(
                "\n".join(
                    [
                        f"[CHUNK_ID={c.chunk_id}]",
                        f"FILE={c.source_file}",
                        f"SECTION={c.section_path or '-'}",
                        c.text,
                    ]
                )
            )
        return "\n\n".join(parts)

    def _build_citations(self, chunks: List[Chunk], selected_ids: List[int]) -> List[Citation]:
        by_id: Dict[int, Chunk] = {c.chunk_id: c for c in chunks}
        out: List[Citation] = []
        for cid in selected_ids:
            c = by_id.get(cid)
            if not c:
                continue
            out.append(
                Citation(
                    chunk_id=c.chunk_id,
                    source_file=c.source_file,
                    section_path=c.section_path,
                    snippet=c.text[:260],
                )
            )
        return out

    def answer(self, question: str, chunks: List[Chunk]) -> AnswerResult:
        logger.info("Generation started | query=%s | chunks=%s", question, len(chunks))
        if not chunks:
            logger.warning("Generation skipped: empty context")
            return AnswerResult(
                question=question,
                answer="Недостаточно информации в документах.",
                citations=[],
                retrieved_chunk_ids=[],
            )

        if self.use_llm and self.llm is not None:
            logger.info("Generation mode: LLM JSON")
            messages = [
                SystemMessage(
                    content=(
                        "Ты отвечаешь по нормативным документам. "
                        "Используй только предоставленный контекст. "
                        "Ответ должен быть кратким и фактологическим. "
                        "Не копируй длинные фрагменты контекста. "
                        "Если факта нет в контексте, напиши 'Недостаточно информации'. "
                        "Для числовых вопросов обязательно укажи число/пункт/таблицу."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Контекст:\n{self._context(chunks)}\n\n"
                        f"Вопрос: {question}\n\n"
                        "Верни JSON по схеме: "
                        "{answer_short: <=220 символов, citation_chunk_ids: [id...]}."
                    )
                ),
            ]
            raw = self.llm.invoke(messages, format=GenerationOutput.model_json_schema()).content
            parsed = GenerationOutput.model_validate_json(raw)
            logger.debug("LLM raw output: %s", raw)
        else:
            logger.info("Generation mode: extractive fallback")
            parsed = self._extractive_generate(question, chunks)

        parsed = self._normalize_output(question, chunks, parsed)

        citations = self._build_citations(chunks, parsed.citation_chunk_ids)
        if not citations and chunks:
            citations = self._build_citations(chunks, [chunks[0].chunk_id])
        source_rows = [
            f"- {c.source_file} | {c.section_path or 'section: n/a'} | chunk_id={c.chunk_id}"
            for c in citations
        ]
        source_text = "\n".join(source_rows) if source_rows else "- источники не определены"

        final_answer = f"{parsed.answer_short}\n\nИсточники:\n{source_text}"
        logger.info(
            "Generation finished | citation_ids=%s | answer_preview=%s",
            parsed.citation_chunk_ids,
            parsed.answer_short[:120],
        )

        return AnswerResult(
            question=question,
            answer=final_answer,
            citations=citations,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
        )

    def _extractive_generate(self, question: str, chunks: List[Chunk]) -> GenerationOutput:
        logger.debug("Extractive generation started")
        query_terms = set(re.findall(r"[A-Za-zА-Яа-я0-9_-]+", question.lower()))
        best_score = -1
        best_sentence = "Недостаточно информации."
        best_chunk_id = chunks[0].chunk_id

        for c in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", c.text)
            for sentence in sentences:
                terms = set(re.findall(r"[A-Za-zА-Яа-я0-9_-]+", sentence.lower()))
                if not terms:
                    continue
                overlap = len(query_terms & terms)
                score = overlap / max(1, len(query_terms))
                if re.search(r"\d", sentence):
                    score += 0.12
                if re.search(r"\b(пункт|таблица|рисунок|раздел|статья)\b", sentence.lower()):
                    score += 0.12
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
                    best_chunk_id = c.chunk_id

        best_sentence = re.sub(r"\s+", " ", best_sentence).strip()
        if len(best_sentence) > 320:
            best_sentence = best_sentence[:317].rstrip() + "..."

        return GenerationOutput(
            answer_short=best_sentence if best_score > 0 else "Недостаточно информации.",
            citation_chunk_ids=[best_chunk_id],
        )

    def _normalize_output(self, question: str, chunks: List[Chunk], parsed: GenerationOutput) -> GenerationOutput:
        answer = self._shorten_answer(parsed.answer_short)
        ids = [cid for cid in parsed.citation_chunk_ids if any(c.chunk_id == cid for c in chunks)]
        if not ids and chunks:
            ids = [chunks[0].chunk_id]

        constraints = self._question_constraints(question)
        if not self._is_fact_sufficient(question, answer):
            fallback = self._extractive_generate(question, chunks)
            answer = self._shorten_answer(fallback.answer_short)
            ids = fallback.citation_chunk_ids or ids
            logger.info("Answer post-check fallback applied")
        elif not self._is_answer_supported_by_chunks(answer, chunks, ids, constraints):
            fallback = self._extractive_generate(question, chunks)
            answer = self._shorten_answer(fallback.answer_short)
            ids = fallback.citation_chunk_ids or ids
            logger.info("Answer support-check fallback applied")

        return GenerationOutput(answer_short=answer, citation_chunk_ids=ids)

    @staticmethod
    def _shorten_answer(text: str) -> str:
        t = re.sub(r"\s+", " ", (text or "").strip())
        if not t:
            return "Недостаточно информации."
        sentences = re.split(r"(?<=[.!?])\s+", t)
        short = " ".join(sentences[:2]).strip()
        if len(short) > 240:
            short = short[:237].rstrip() + "..."
        return short

    @staticmethod
    def _is_fact_sufficient(question: str, answer: str) -> bool:
        q = question.lower()
        a = (answer or "").lower()
        if not a or a == "недостаточно информации.":
            return False

        constraints = AnswerGenerator._question_constraints(q)

        ok = True
        if constraints["needs_number"]:
            ok = ok and bool(re.search(r"\d", a))
        if constraints["needs_point"]:
            ok = ok and bool(re.search(r"(пункт|п\.)\s*\d+(\.\d+)*", a))
        if constraints["needs_table"]:
            ok = ok and ("таблиц" in a or re.search(r"\b\d+(\.\d+)*\b", a) is not None)
        if constraints["needs_units"]:
            ok = ok and bool(re.search(r"(мм|км/ч|кмч|°c|с|см|м\b)", a))
        return ok

    @staticmethod
    def _question_constraints(question: str) -> Dict[str, bool]:
        q = (question or "").lower()
        return {
            "needs_number": any(x in q for x in ("сколько", "максим", "миним", "температур", "скорост", "коэффициент", "толщин", "в каких зонах")),
            "needs_point": "в каком пункте" in q or "пункт" in q,
            "needs_table": "таблиц" in q or "табл" in q,
            "needs_units": any(x in q for x in ("скорост", "температур", "толщин")),
        }

    @staticmethod
    def _is_answer_supported_by_chunks(
        answer: str,
        chunks: List[Chunk],
        selected_ids: List[int],
        constraints: Dict[str, bool],
    ) -> bool:
        if not chunks:
            return False
        selected = [c for c in chunks if c.chunk_id in selected_ids] or chunks[:2]
        combined = " ".join(c.text.lower() for c in selected)
        a = (answer or "").lower()

        answer_numbers = set(re.findall(r"\d+(?:[.,]\d+)?(?:/\d+)?", a))
        if constraints["needs_number"] and not answer_numbers:
            return False
        if answer_numbers:
            present = sum(1 for n in answer_numbers if n in combined)
            if present == 0:
                return False

        if constraints["needs_point"] and not re.search(r"(пункт|п\.)\s*\d+(\.\d+)*", a):
            return False
        if constraints["needs_table"] and "таблиц" not in a and "табл" not in a:
            return False
        return True
