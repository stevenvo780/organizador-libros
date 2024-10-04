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
import pytesseract
from PIL import Image
import tempfile
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
            stderr_output = buf.getvalue()
            if stderr_output:
                log_error(ruta_archivo, f"PyMuPDF stderr: {stderr_output}")
            for warning in w:
                log_error(ruta_archivo, f"PyMuPDF warning: {warning.message}")
        if texto.strip():
            return clean_text(texto), None
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
            stderr_output = buf.getvalue()
            if stderr_output:
                log_error(ruta_archivo, f"PyPDF2 stderr: {stderr_output}")
            for warning in w:
                log_error(ruta_archivo, f"PyPDF2 warning: {warning.message}")
        if texto.strip():
            return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing PDF with PyPDF2: {e}")

    try:
        texto = pdfminer_extract_text(ruta_archivo, maxpages=MAX_PAGES)
        if texto.strip():
            return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing PDF with pdfminer.six: {e}")

    try:
        documento = fitz.open(ruta_archivo)
        texto = ""
        for num_pagina in range(min(MAX_PAGES, len(documento))):
            pagina = documento.load_page(num_pagina)
            imagen = pagina.get_pixmap()
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_img_file:
                imagen.save(temp_img_file.name)
                texto_ocr = pytesseract.image_to_string(Image.open(temp_img_file.name))
                texto += texto_ocr + "\n"
        if texto.strip():
            return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing PDF with OCR: {e}")

    log_error(ruta_archivo, "No se pudo extraer el texto del PDF.")
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
        for warning in w:
            log_error(ruta_archivo, f"EPUB warning: {warning.message}")
        return clean_text(texto), None
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
        for warning in w:
            log_error(ruta_archivo, f"DOCX warning: {warning.message}")
        return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing DOCX: {e}")
        return None, None

def process_doc(ruta_archivo):
    try:
        texto = pypandoc.convert_file(ruta_archivo, 'plain', format='doc')
        return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing DOC: {e}")
        return None, None

def process_rtf(ruta_archivo):
    try:
        texto = pypandoc.convert_file(ruta_archivo, 'plain', format='rtf')
        return clean_text(texto), None
    except Exception as e:
        log_error(ruta_archivo, f"Error processing RTF: {e}")
        return None, None

def process_file(ruta_archivo, ext):
    if ext in ['.pdf', '.PDF', '.pdf_']:
        result, author = process_pdf(ruta_archivo)
    elif ext == '.epub':
        result, author = process_epub(ruta_archivo)
    elif ext == '.docx':
        result, author = process_docx(ruta_archivo)
    elif ext in ['.doc', '.DOC']:
        result, author = process_doc(ruta_archivo)
    elif ext == '.rtf':
        result, author = process_rtf(ruta_archivo)
    else:
        log_error(ruta_archivo, f"Unsupported file type: {ext}")
        result, author = None, None
    return result, author
