import fitz
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
import warnings
import docx
from ebooklib import epub
import re
import io
from contextlib import redirect_stderr
import pypandoc
from utils import clean_text, log_error

MAX_PAGES = 10
MAX_PARAGRAPHS_PER_PAGE = 30
MAX_EPUB_ITEMS = 50

def process_pdf(ruta_archivo):
    # Primero intentamos con pdfminer.six (generalmente más rápido)
    try:
        with io.StringIO() as buf, redirect_stderr(buf):
            texto = pdfminer_extract_text(ruta_archivo, maxpages=MAX_PAGES)
            stderr_output = buf.getvalue()
            if stderr_output:
                log_error(ruta_archivo, f"pdfminer.six stderr: {stderr_output}")
        if texto.strip():
            return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing PDF with pdfminer.six: {e}")

    # Intentamos con PyMuPDF (fitz)
    try:
        with io.StringIO() as buf, redirect_stderr(buf):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("ignore")
                documento = fitz.open(ruta_archivo)
                texto = ""
                for num_pagina in range(min(MAX_PAGES, len(documento))):
                    pagina = documento.load_page(num_pagina)
                    texto += pagina.get_text() + "\n"
                stderr_output = buf.getvalue()
                if stderr_output:
                    log_error(ruta_archivo, f"PyMuPDF stderr: {stderr_output}")
                for warning in w:
                    log_error(ruta_archivo, f"PyMuPDF warning: {warning.message}")
        if texto.strip():
            return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing PDF with PyMuPDF: {e}")

    # Intentamos con PyPDF2
    try:
        with io.StringIO() as buf, redirect_stderr(buf):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("ignore")
                lector = PdfReader(ruta_archivo)
                num_paginas = min(MAX_PAGES, len(lector.pages))
                texto = ''
                for num_pagina in range(num_paginas):
                    pagina = lector.pages[num_pagina]
                    texto_pagina = pagina.extract_text()
                    if texto_pagina:
                        texto += texto_pagina + '\n'
                stderr_output = buf.getvalue()
                if stderr_output:
                    log_error(ruta_archivo, f"PyPDF2 stderr: {stderr_output}")
                for warning in w:
                    log_error(ruta_archivo, f"PyPDF2 warning: {warning.message}")
        if texto.strip():
            return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing PDF with PyPDF2: {e}")

    log_error(ruta_archivo, "No se pudo extraer el texto del PDF.")
    return None, None

def process_epub(ruta_archivo):
    try:
        with io.StringIO() as buf, redirect_stderr(buf):
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
                stderr_output = buf.getvalue()
                if stderr_output:
                    log_error(ruta_archivo, f"EPUB stderr: {stderr_output}")
                for warning in w:
                    log_error(ruta_archivo, f"EPUB warning: {warning.message}")
        return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing EPUB: {e}")
        return None, None

def process_docx(ruta_archivo):
    try:
        with io.StringIO() as buf, redirect_stderr(buf):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                documento = docx.Document(ruta_archivo)
                texto = ''
                num_parrafos = min(MAX_PAGES * MAX_PARAGRAPHS_PER_PAGE, len(documento.paragraphs))
                for i in range(num_parrafos):
                    texto += documento.paragraphs[i].text + '\n'
                stderr_output = buf.getvalue()
                if stderr_output:
                    log_error(ruta_archivo, f"DOCX stderr: {stderr_output}")
                for warning in w:
                    log_error(ruta_archivo, f"DOCX warning: {warning.message}")
        return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing DOCX: {e}")
        return None, None

def process_doc(ruta_archivo):
    try:
        with io.StringIO() as buf, redirect_stderr(buf):
            texto = pypandoc.convert_file(ruta_archivo, 'plain', format='doc')
            stderr_output = buf.getvalue()
            if stderr_output:
                log_error(ruta_archivo, f"DOC stderr: {stderr_output}")
        return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing DOC: {e}")
        return None, None

def process_rtf(ruta_archivo):
    try:
        with io.StringIO() as buf, redirect_stderr(buf):
            texto = pypandoc.convert_file(ruta_archivo, 'plain', format='rtf')
            stderr_output = buf.getvalue()
            if stderr_output:
                log_error(ruta_archivo, f"RTF stderr: {stderr_output}")
        return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing RTF: {e}")
        return None, None

def process_file(ruta_archivo, ext):
    if ext in ['.pdf', '.PDF', '.pdf_']:
        return process_pdf(ruta_archivo)
    elif ext == '.epub':
        return process_epub(ruta_archivo)
    elif ext == '.docx':
        return process_docx(ruta_archivo)
    elif ext in ['.doc', '.DOC']:
        return process_doc(ruta_archivo)
    elif ext == '.rtf':
        return process_rtf(ruta_archivo)
    else:
        log_error(ruta_archivo, f"Unsupported file type: {ext}")
        return None, None