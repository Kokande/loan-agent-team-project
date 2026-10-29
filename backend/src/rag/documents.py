"""Загрузка документов и чанкинг с сохранением страниц источника."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Page:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    page: int | None = None


def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[^\S\n]+", " ", text).strip()


def load_pages(path: str | Path) -> list[Page]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        pages = [Page(path.read_text(encoding="utf-8-sig"))]
    elif suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = [Page(page.extract_text() or "", i) for i, page in enumerate(reader.pages, 1)]
    elif suffix == ".docx":
        from docx import Document
        from docx.table import Table

        parts = []
        for block in Document(path).iter_inner_content():
            if isinstance(block, Table):
                parts.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
            else:
                parts.append(block.text)
        pages = [Page("\n\n".join(parts))]
    else:
        raise ValueError(f"Неподдерживаемый формат {suffix!r}: нужны TXT, MD, PDF или DOCX")
    pages = [Page(_clean(page.text), page.page) for page in pages if _clean(page.text)]
    if not pages:
        raise ValueError(
            "Документ не содержит текста. Для сканированного PDF сначала выполните OCR."
        )
    return pages


def chunk_pages(
    pages: list[Page], chunk_size: int = 1600, overlap: int = 200,
) -> list[Chunk]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Нужно chunk_size > 0 и 0 <= overlap < chunk_size")
    chunks = []
    for page in pages:
        text = page.text.strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Не создаём короткие чанки из-за границы в самом начале окна.
                floor = start + max(chunk_size // 2, overlap + 1)
                for separator in ("\n\n", "\n", ". ", "; ", " "):
                    boundary = text.rfind(separator, floor, end)
                    if boundary != -1:
                        end = boundary + len(separator)
                        break
            fragment = text[start:end].strip()
            if fragment:
                chunks.append(Chunk(len(chunks), fragment, page.page))
            if end == len(text):
                break
            start = end - overlap
    return chunks
