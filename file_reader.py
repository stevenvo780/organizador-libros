import fitz
from PyPDF2 import PdfReader
import warnings
import docx
from ebooklib import epub
import re
import io
from contextlib import redirect_stderr
from utils import clean_text, log_error

MAX_PAGES = 10
MAX_PARAGRAPHS_PER_PAGE = 30
MAX_EPUB_ITEMS = 10

def process_pdf(ruta_archivo):
    try:
        with io.StringIO() as buf, redirect_stderr(buf):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("ignore")
                documento = fitz.open(ruta_archivo)
                texto = ""
                for num_pagina in range(min(MAX_PAGES, len(documento))):
                    pagina = documento.load_page(num_pagina)
                    texto += pagina.get_text() + "\n"
                info = documento.metadata
                author = info.get('author', None) if info else None
            stderr_output = buf.getvalue()
            if stderr_output:
                log_error(ruta_archivo, f"PyMuPDF stderr: {stderr_output}")
            for warning in w:
                log_error(ruta_archivo, f"PyMuPDF warning: {warning.message}")
        return clean_text(texto), author
    except Exception as e:
        log_error(ruta_archivo, f"Error processing PDF with PyMuPDF: {e}")

    try:
        with io.StringIO() as buf, redirect_stderr(buf):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("ignore")
                lector = PdfReader(ruta_archivo)
                num_paginas = min(MAX_PAGES, len(lector.pages))
                texto = ''
                for num_pagina in range(num_paginas):
                    pagina = lector.pages[num_pagina]
                    try:
                        texto_pagina = pagina.extract_text()
                        if texto_pagina:
                            texto += texto_pagina + '\n'
                    except Exception as e:
                        log_error(ruta_archivo, f"Error extracting text from page {num_pagina}: {e}")
                        continue
                info = lector.metadata
                author = info.get('/Author', None) if info else None
            stderr_output = buf.getvalue()
            if stderr_output:
                log_error(ruta_archivo, f"PyPDF2 stderr: {stderr_output}")
            for warning in w:
                log_error(ruta_archivo, f"PyPDF2 warning: {warning.message}")
        return clean_text(texto), author
    except Exception as e:
        log_error(ruta_archivo, f"Error processing PDF with PyPDF2: {e}")
        return None, None

def process_epub(ruta_archivo):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            libro = epub.read_epub(ruta_archivo)
            texto = ''
            conteo = 0
            for item in libro.get_items():
                if item.get_type() == epub.EpubHtml:
                    contenido = item.get_content().decode('utf-8', errors='ignore')
                    contenido = re.sub(r'<[^>]+>', '', contenido)
                    texto += contenido + '\n'
                    conteo += 1
                    if conteo >= MAX_EPUB_ITEMS:
                        break
            authors = libro.get_metadata('DC', 'creator')
            author = authors[0][0] if authors else None
        for warning in w:
            log_error(ruta_archivo, f"EPUB warning: {warning.message}")
        return clean_text(texto), author
    except Exception as e:
        log_error(ruta_archivo, f"Error processing EPUB: {e}")
        return None, None

def process_docx(ruta_archivo):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            documento = docx.Document(ruta_archivo)
            texto = ''
            num_parrafos = min(MAX_PAGES * MAX_PARAGRAPHS_PER_PAGE, len(documento.paragraphs))
            for i in range(num_parrafos):
                texto += documento.paragraphs[i].text + '\n'
            core_properties = documento.core_properties
            author = core_properties.author
        for warning in w:
            log_error(ruta_archivo, f"DOCX warning: {warning.message}")
        return clean_text(texto), author
    except Exception as e:
        log_error(ruta_archivo, f"Error processing DOCX: {e}")
        return None, None

def process_file(ruta_archivo, ext):
    if ext == '.pdf':
        result, author = process_pdf(ruta_archivo)
    elif ext == '.epub':
        result, author = process_epub(ruta_archivo)
    elif ext == '.docx':
        result, author = process_docx(ruta_archivo)
    else:
        log_error(ruta_archivo, f"Unsupported file type: {ext}")
        result, author = None, None
    return result, author
