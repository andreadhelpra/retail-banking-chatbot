from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from app.config import MISTRAL_EMBED_MODEL, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class FAQRetriever:
    def __init__(self, mistral_client, faq_dir: str | None = None):
        self._client = mistral_client
        if faq_dir is None:
            faq_dir = str(Path(__file__).parent.parent.parent / "data" / "faqs")
        self.chunks: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._load_and_embed(faq_dir)

    def _load_and_embed(self, faq_dir: str) -> None:
        faq_path = Path(faq_dir)
        if not faq_path.exists():
            logger.error(f"FAQ directory not found: {faq_dir}")
            return
        for md_file in sorted(faq_path.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            file_chunks = self._chunk_markdown(text, source=md_file.name)
            self.chunks.extend(file_chunks)
        if not self.chunks:
            logger.warning("No FAQ chunks loaded")
            return
        logger.info(f"[RETRIEVER] Loaded {len(self.chunks)} chunks from {faq_dir}")
        self._embeddings = self._embed_texts(self.chunks)
        logger.info(f"[RETRIEVER] Embedded {len(self.chunks)} chunks")

    def _chunk_markdown(self, text: str, source: str = "") -> list[str]:
        lines = text.strip().split("\n")
        title = ""
        chunks: list[str] = []
        current_chunk_lines: list[str] = []
        for line in lines:
            if re.match(r"^## ", line):
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                current_chunk_lines = [title, line] if title else [line]
            elif re.match(r"^# ", line):
                title = line
                current_chunk_lines.append(line)
            else:
                current_chunk_lines.append(line)
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)
        return chunks

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(model=MISTRAL_EMBED_MODEL, inputs=batch)
            for item in response.data:
                all_embeddings.append(item.embedding)
        return np.array(all_embeddings)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if self._embeddings is None or len(self.chunks) == 0:
            return []
        query_embedding = self._embed_texts([query])[0]
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = np.dot(self._embeddings, query_embedding) / np.maximum(norms, 1e-10)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= SIMILARITY_THRESHOLD:
                results.append((self.chunks[idx], score))
        return results
