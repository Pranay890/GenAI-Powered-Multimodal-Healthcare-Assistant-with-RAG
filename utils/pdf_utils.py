from pypdf import PdfReader


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])
