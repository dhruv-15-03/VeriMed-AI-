import os
import pdfplumber
from docx import Document

def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        print(f"[PDF Error] {e}")
        return None

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"[DOCX Error] {e}")
        return None

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"[TXT Error] {e}")
        return None

def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        print("Unsupported file type.")
        return None

if __name__ == "__main__":
    file_path = input("Enter path to medical report (PDF/DOCX/TXT): ")
    content = extract_text(file_path)
    if content:
        print("\n--- Extracted Text ---\n")
        print(content[:3000])  # Show only first 3000 chars
