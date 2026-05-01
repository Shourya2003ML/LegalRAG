import os
import fitz
from typing import cast

def load_pdfs_from_folder(folder_path: str):
    documents = []
    if not os.path.exists(folder_path):
        raise ValueError(f"Path Not Found: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += cast("str", page.get_text("text", sort=True))
                documents.append({"text": text, "filename": filename})
                doc.close()
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return documents

def chunk_text(text:str, chunk_size: int = 500, overlap: int = 50):
    """Chunking of text with overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start+chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks