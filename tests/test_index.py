import numpy as np
import pytest

from trustqa.config import load_config
from trustqa.processing.process import process_pdfs
from trustqa.retrieval.index import SearchIndex, build_index
from trustqa.schemas import Chunk


@pytest.fixture
def cfg(tmp_path):
    env = tmp_path / ".env"
    env.write_text("")
    return load_config(env, chunk_size=200, chunk_overlap=40, indexes_dir=tmp_path / "idx")


@pytest.fixture
def chunks(solar_pdf, river_pdf, cfg):
    return process_pdfs([solar_pdf, river_pdf], cfg)


def test_vectors_are_normalised_and_aligned(chunks, cfg, fake_embedder):
    idx = build_index(chunks, cfg)
    assert len(idx) == idx.index.ntotal == len(chunks)
    vecs = idx.index.reconstruct_n(0, len(idx))
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    # vector i is the embedding of chunks[i]
    for i in (0, len(chunks) - 1):
        expected = fake_embedder.encode([chunks[i].text])[0]
        assert np.allclose(vecs[i], expected, atol=1e-5)
    assert idx.chunks is not None and [c.chunk_id for c in idx.chunks] == [
        c.chunk_id for c in chunks]


def test_save_load_round_trip(chunks, cfg, fake_embedder, tmp_path):
    idx = build_index(chunks, cfg)
    idx.save(tmp_path / "saved")
    loaded = SearchIndex.load(tmp_path / "saved", embedder=fake_embedder)
    assert loaded.chunks == idx.chunks
    assert loaded.embedding_model == cfg.embedding_model
    q = "installed capacity of the Pavagada plant"
    assert loaded.search(q, 3) == idx.search(q, 3)


def test_load_detects_misalignment(chunks, cfg, fake_embedder, tmp_path):
    build_index(chunks, cfg).save(tmp_path / "saved")
    lines = (tmp_path / "saved" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "saved" / "chunks.jsonl").write_text("\n".join(lines[::-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        SearchIndex.load(tmp_path / "saved")


def test_empty_and_duplicate_rejected(chunks, cfg, fake_embedder):
    with pytest.raises(ValueError, match="zero"):
        build_index([], cfg)
    dup = Chunk.from_dict(chunks[0].to_dict())
    with pytest.raises(ValueError, match="duplicate"):
        build_index([chunks[0], dup], cfg)


def test_search_caps_top_k(chunks, cfg, fake_embedder):
    idx = build_index(chunks, cfg)
    res = idx.search("dam", top_k=1000)
    assert len(res) == len(chunks)
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
