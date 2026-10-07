from dataclasses import dataclass


@dataclass(frozen=True)
class ParentChildChunk:
    parent_index: int
    child_index: int
    parent_text: str
    child_text: str


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def split_parent_child(
    text: str,
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
) -> list[ParentChildChunk]:
    if child_size > parent_size:
        raise ValueError("child size must not exceed parent size")

    chunks = []
    parents = split_text(text, parent_size, parent_overlap)
    for parent_index, parent_text in enumerate(parents):
        children = split_text(parent_text, child_size, child_overlap)
        chunks.extend(
            ParentChildChunk(
                parent_index=parent_index,
                child_index=child_index,
                parent_text=parent_text,
                child_text=child_text,
            )
            for child_index, child_text in enumerate(children)
        )
    return chunks
