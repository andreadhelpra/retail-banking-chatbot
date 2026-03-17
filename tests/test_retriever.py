import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from app.services.retriever import FAQRetriever


@pytest.fixture
def mock_mistral_client():
    client = MagicMock()
    def fake_embed(model, inputs):
        response = MagicMock()
        embeddings = []
        for _ in inputs:
            vec = np.random.RandomState(42).rand(1024).tolist()
            emb = MagicMock()
            emb.embedding = vec
            embeddings.append(emb)
        response.data = embeddings
        return response
    client.embeddings.create = fake_embed
    return client


def test_load_faqs(mock_mistral_client, tmp_path):
    faq_dir = tmp_path / "faqs"
    faq_dir.mkdir()
    (faq_dir / "test.md").write_text("# Test\n\nThis is a test FAQ about opening hours.\n\n## Section\n\nMore details here.")
    retriever = FAQRetriever(mock_mistral_client, str(faq_dir))
    assert len(retriever.chunks) > 0


def test_chunk_splitting(mock_mistral_client, tmp_path):
    faq_dir = tmp_path / "faqs"
    faq_dir.mkdir()
    (faq_dir / "test.md").write_text("# Title\n\nFirst section content.\n\n## Second\n\nSecond section content.")
    retriever = FAQRetriever(mock_mistral_client, str(faq_dir))
    assert len(retriever.chunks) >= 2


def test_search_returns_results(mock_mistral_client, tmp_path):
    faq_dir = tmp_path / "faqs"
    faq_dir.mkdir()
    (faq_dir / "test.md").write_text("# Hours\n\nWe are open 9-5.\n\n## Weekend\n\nClosed on Sunday.")
    retriever = FAQRetriever(mock_mistral_client, str(faq_dir))
    results = retriever.search("opening hours", top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    for chunk, score in results:
        assert isinstance(chunk, str)
        assert isinstance(score, float)
