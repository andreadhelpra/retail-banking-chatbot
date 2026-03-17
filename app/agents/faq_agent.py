from __future__ import annotations

import logging
from typing import Any

from app.config import MISTRAL_SMALL_MODEL
from app.services.retriever import FAQRetriever

logger = logging.getLogger(__name__)

FAQ_SYSTEM_PROMPT = """You are the FAQ agent for BNP Paribas retail banking assistant.
Your job is to answer customer questions using ONLY the information provided in the context chunks below.

Rules:
- Only use information from the provided context to answer the question.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and helpful.
- Use specific numbers, dates, and details from the context when available.
- Do not make up information that is not in the context.
- Format your answer in a clear, readable way.

Context chunks:
{context}
"""


async def handle_faq(
    mistral_client,
    retriever: FAQRetriever,
    message: str,
) -> dict[str, Any]:
    """Handle an FAQ query using RAG with Mistral Small."""

    results = retriever.search(message, top_k=3)

    if not results:
        logger.info("[FAQ] No relevant chunks found, returning fallback")
        return {
            "response": "I don't have information on that topic. Would you like me to connect you with an advisor?",
            "retrieved_chunks": [],
        }

    chunks = [chunk for chunk, score in results]
    scores = [score for chunk, score in results]

    logger.info(f"[FAQ] Retrieved {len(chunks)} chunks with scores: {[f'{s:.3f}' for s in scores]}")

    context = "\n\n---\n\n".join(f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks))

    messages = [
        {"role": "system", "content": FAQ_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": message},
    ]

    try:
        response = await mistral_client.chat.complete_async(
            model=MISTRAL_SMALL_MODEL,
            messages=messages,
        )

        answer = response.choices[0].message.content

        logger.info(f"[FAQ] Generated answer ({len(answer)} chars)")

        return {
            "response": answer,
            "retrieved_chunks": chunks,
        }

    except Exception as e:
        logger.error(f"[FAQ] Error generating answer: {e}")
        return {
            "response": "I'm experiencing a temporary issue. Please try again in a moment.",
            "retrieved_chunks": chunks,
        }
