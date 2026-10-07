import pytest

from app.services.chunking import split_parent_child, split_text


def test_split_text_creates_overlapping_chunks():
    chunks = split_text("alpha beta gamma delta epsilon zeta", chunk_size=18, overlap=5)
    assert len(chunks) > 1
    assert chunks[0].split()[-1] in chunks[1].split()[:2]


def test_split_text_handles_empty_input():
    assert split_text(" \n ", chunk_size=100, overlap=10) == []


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=10, overlap=10)


def test_split_parent_child_keeps_child_chunks_with_parent_context():
    chunks = split_parent_child(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        parent_size=32,
        parent_overlap=4,
        child_size=18,
        child_overlap=3,
    )

    assert len(chunks) > 2
    assert chunks[0].parent_index == 0
    assert chunks[0].child_index == 0
    assert chunks[0].child_text in chunks[0].parent_text
    assert all(chunk.child_text in chunk.parent_text for chunk in chunks)
    assert {chunk.parent_index for chunk in chunks} == {0, 1, 2}


def test_split_parent_child_rejects_child_larger_than_parent():
    with pytest.raises(ValueError, match="child size"):
        split_parent_child(
            "alpha beta gamma", parent_size=10, parent_overlap=2, child_size=11, child_overlap=2
        )
