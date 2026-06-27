from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_path: str) -> list[dict]:
    """
    Reads a PDF and returns a list of chunks with metadata.
    Each chunk is small enough for the LLM context window
    but large enough to contain meaningful information.
    """
    path   = Path(pdf_path)
    reader = PdfReader(str(path))

    # Extract text page by page, keeping track of page numbers
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text":      text.strip(),
                "page":      page_num,
                "source":    path.name,
            })

    if not pages:
        raise ValueError(f"No text found in {path.name}. Is it a scanned PDF?")

    print(f"Loaded {path.name} — {len(pages)} pages")

    # Split into overlapping chunks so context isn't lost at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # ~600 words per chunk
        chunk_overlap=100,     # 100 chars overlap between chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page_data in pages:
        splits = splitter.split_text(page_data["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text":   split,
                "page":   page_data["page"],
                "source": page_data["source"],
                "chunk":  i,
            })

    print(f"Split into {len(chunks)} chunks")
    return chunks


def load_text(text: str, source_name: str = "manual_input") -> list[dict]:
    """
    Same as load_pdf but for raw text input.
    Used when student types notes directly instead of uploading PDF.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    splits = splitter.split_text(text)
    return [
        {"text": s, "page": 1, "source": source_name, "chunk": i}
        for i, s in enumerate(splits)
    ]