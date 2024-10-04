import os
import shutil
import re
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login
import torch
from PyPDF2 import PdfReader
from ebooklib import epub
import docx
import unicodedata
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
login(token=token)

#CARPETA_ENTRADA = '/mnt/FASTDATA/LibrosBiblioteca'
CARPETA_ENTRADA = 'Libros'
CARPETA_SALIDA = 'Libros_Organizados'
LOG_FILE = 'errores_procesamiento.json'
MAX_WORKERS = os.cpu_count()
MAX_PAGES = 10
MAX_PARAGRAPHS_PER_PAGE = 30
MAX_EPUB_ITEMS = 10
MAX_CHARACTERS = 15000
BATCH_SIZE = 64
QUESTION_AUTHOR = "¿Quién es el autor del libro?"

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True
if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.9)
else:
    torch.set_num_threads(os.cpu_count())

tqa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    device=0 if device == "cuda" else -1
)

ner_pipeline = pipeline("ner", model="dccuchile/bert-base-spanish-wwm-cased-finetuned-ner")

known_authors = set()
log_data = {
    "archivos_error": [],
    "archivos_no_soportados": []
}

def log_error(ruta_archivo, mensaje):
    log_data["archivos_error"].append({"archivo": ruta_archivo, "error": mensaje})

def clean_text(text):
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_author_name(name):
    name = name.lower()
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = re.sub(r'[^a-z\s]', '', name)
    name = ' '.join(name.split())
    return name

def get_best_matching_author(name):
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    best_match = None
    highest_similarity = 0.0
    for known_author in known_authors:
        sim = similarity(name, known_author)
        if sim > highest_similarity:
            highest_similarity = sim
            best_match = known_author

    if highest_similarity > 0.8:
        return best_match
    else:
        known_authors.add(name)
        return name

def extract_author_using_ner(text):
    ner_results = ner_pipeline(text)
    author_candidates = [entity['word'] for entity in ner_results if entity['entity'] == 'B-PER']
    if author_candidates:
        return ' '.join(author_candidates)
    return None

def process_pdf(ruta_archivo):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lector = PdfReader(ruta_archivo)
            for warning in w:
                log_error(ruta_archivo, str(warning.message))
        num_paginas = min(MAX_PAGES, len(lector.pages))
        texto = ''
        for num_pagina in range(num_paginas):
            pagina = lector.pages[num_pagina]
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto += texto_pagina + '\n'
        info = lector.metadata
        author = info.get('/Author', None) if info else None
        return clean_text(texto), author
    except Exception as e:
        log_error(ruta_archivo, f"Error processing PDF: {e}")
        return None, None

def process_epub(ruta_archivo):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            libro = epub.read_epub(ruta_archivo)
            for warning in w:
                log_error(ruta_archivo, str(warning.message))
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
        return clean_text(texto), author
    except Exception as e:
        log_error(ruta_archivo, f"Error processing EPUB: {e}")
        return None, None

def process_docx(ruta_archivo):
    try:
        documento = docx.Document(ruta_archivo)
        texto = ''
        num_parrafos = min(MAX_PAGES * MAX_PARAGRAPHS_PER_PAGE, len(documento.paragraphs))
        for i in range(num_parrafos):
            texto += documento.paragraphs[i].text + '\n'
        core_properties = documento.core_properties
        author = core_properties.author
        return clean_text(texto), author
    except Exception as e:
        log_error(ruta_archivo, f"Error processing DOCX: {e}")
        return None, None

def process_file(args):
    ruta_archivo, ext = args
    if ext == '.pdf':
        result, error = process_pdf(ruta_archivo)
    elif ext == '.epub':
        result, error = process_epub(ruta_archivo)
    elif ext == '.docx':
        result, error = process_docx(ruta_archivo)
    else:
        log_error(ruta_archivo, f"Unsupported file type: {ext}")
        result = None
    return ruta_archivo, result, error

def organize_file(args):
    idx, author, rutas_archivos = args
    ruta_archivo = rutas_archivos[idx]
    nombre_archivo = os.path.basename(ruta_archivo)
    try:
        if not author or author.strip() == '' or author.lower() == 'no answer':
            nombre_autor = 'Autor Desconocido'
        else:
            nombre_autor = normalize_author_name(author)
            nombre_autor = get_best_matching_author(nombre_autor)
            nombre_autor = re.sub(r'[<>:"/\\|?*]', '', nombre_autor)

        carpeta_autor = os.path.join(CARPETA_SALIDA, nombre_autor)

        if not os.path.exists(carpeta_autor):
            os.makedirs(carpeta_autor)

        destino = os.path.join(carpeta_autor, nombre_archivo)
        if not os.path.exists(destino):
            shutil.copy2(ruta_archivo, destino)
    except Exception as e:
        log_error(ruta_archivo, str(e))

def main():
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)

    archivos_para_procesar = []
    for root, _, files in os.walk(CARPETA_ENTRADA):
        for nombre_archivo in files:
            ruta_archivo = os.path.join(root, nombre_archivo)
            if os.path.isfile(ruta_archivo):
                ext = os.path.splitext(ruta_archivo)[1].lower()
                if ext in ['.pdf', '.epub', '.docx']:
                    archivos_para_procesar.append((ruta_archivo, ext))
                else:
                    log_data["archivos_no_soportados"].append(ruta_archivo)

    textos_para_procesar = []
    rutas_archivos = []
    autores_extraidos = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, args): args[0] for args in archivos_para_procesar}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extrayendo textos de archivos", unit="archivo"):
            ruta_archivo = futures[future]
            try:
                ruta_archivo, result, error = future.result()
                if result is not None:
                    textos_para_procesar.append(result)
                    rutas_archivos.append(ruta_archivo)
                    autores_extraidos.append(error)
            except Exception as e:
                log_error(ruta_archivo, str(e))

    indices_sin_autor = [i for i, autor in enumerate(autores_extraidos) if not autor]

    if indices_sin_autor:
        for batch_start in tqdm(range(0, len(indices_sin_autor), BATCH_SIZE), desc="Extrayendo autores", unit="batch"):
            batch_indices = indices_sin_autor[batch_start:batch_start+BATCH_SIZE]
            batch_texts = [textos_para_procesar[i] for i in batch_indices]
            batch_rutas = [rutas_archivos[i] for i in batch_indices]

            qa_inputs = []
            valid_indices = []
            for idx, text in zip(batch_indices, batch_texts):
                context = text[:MAX_CHARACTERS].strip()
                if not context:
                    log_error(rutas_archivos[idx], "Contexto vacío, no se puede extraer autor")
                    continue
                qa_inputs.append({'context': context, 'question': QUESTION_AUTHOR})
                valid_indices.append(idx)

            if not qa_inputs:
                continue

            try:
                outputs = tqa_pipeline(qa_inputs, batch_size=BATCH_SIZE)
                for idx_output, output in zip(valid_indices, outputs):
                    answer = output.get('answer', None)
                    autores_extraidos[idx_output] = answer if answer else extract_author_using_ner(textos_para_procesar[idx_output])
            except Exception as e:
                for idx_error in valid_indices:
                    log_error(rutas_archivos[idx_error], f"Error processing QA: {e}")
                    autores_extraidos[idx_error] = None

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(organize_file, (idx, author, rutas_archivos)) for idx, author in enumerate(autores_extraidos)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Organizando archivos por autor", unit="archivo"):
            future.result()

    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        json.dump(log_data, log_file, indent=4, ensure_ascii=False)

    print("Terminé")

if __name__ == '__main__':
    main()