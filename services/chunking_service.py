import os
import pypdf
import docx                           # python-docx
import email
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingService:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── PUBLIC ───────────────────────────────────────────────────────────────
    def process_file(self, file_path: str, filename: str, doc_id: str) -> list[dict]:
        """
        Master method. Returns list of chunk dicts:
        {
            text, source, page, doc_id, chunk_index, file_type
        }
        """
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            raw = self._process_pdf(file_path, filename)
        elif ext == ".docx":
            raw = self._process_docx(file_path, filename)
        elif ext == ".csv":
            raw = self._process_csv(file_path, filename)
        elif ext in (".eml", ".msg", ".txt"):
            raw = self._process_email(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Attach shared metadata
        for i, chunk in enumerate(raw):
            chunk["doc_id"] = doc_id
            chunk["chunk_index"] = i
            chunk["file_type"] = ext.lstrip(".")

        return raw

    # ── PDF ──────────────────────────────────────────────────────────────────
    def _process_pdf(self, file_path: str, filename: str) -> list[dict]:
        chunks = []
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                for chunk_text in self.splitter.split_text(text):
                    chunk_text = chunk_text.strip()
                    if len(chunk_text) > 30:
                        chunks.append({
                            "text": chunk_text,
                            "source": filename,
                            "page": str(i),       # "3" → "from page 3 of PDF"
                        })
        return chunks

    # ── DOCX ─────────────────────────────────────────────────────────────────
    def _process_docx(self, file_path: str, filename: str) -> list[dict]:
        doc = docx.Document(file_path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [
            {"text": t.strip(), "source": filename, "page": None}
            for t in self.splitter.split_text(full_text)
            if len(t.strip()) > 30
        ]

    # ── CSV ───────────────────────────────────────────────────────────────────
    def _process_csv(self, file_path: str, filename: str) -> list[dict]:
        import pandas as pd
        df = pd.read_csv(file_path)
        chunks = []

        # Column description chunk — always first
        header = f"File: {filename}. Columns: {', '.join(df.columns.tolist())}. Total rows: {len(df)}."
        chunks.append({"text": header, "source": filename, "page": None})

        # Batch rows — 10 rows per chunk
        for i in range(0, len(df), 10):
            batch = df.iloc[i:i + 10]
            text = f"Rows {i + 1} to {i + len(batch)}:\n{batch.to_string(index=False)}"
            chunks.append({"text": text, "source": filename, "page": None})

        return chunks

    # ── EMAIL ─────────────────────────────────────────────────────────────────
    def _process_email(self, file_path: str, filename: str) -> list[dict]:
        with open(file_path, "rb") as f:
            msg = email.message_from_bytes(f.read())

        subject = msg.get("Subject", "No Subject")
        sender = msg.get("From", "Unknown")
        date = msg.get("Date", "Unknown")
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="replace")
        else:
            raw = msg.get_payload(decode=True)
            body = raw.decode(errors="replace") if raw else str(msg.get_payload())

        full_text = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{body}"
        return [
            {"text": t.strip(), "source": filename, "page": None}
            for t in self.splitter.split_text(full_text)
            if len(t.strip()) > 30
        ]
