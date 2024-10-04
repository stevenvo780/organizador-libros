import os
import shutil
import re
from PyPDF2 import PdfReader
from ebooklib import epub
import docx
from transformers import pipeline
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import login
import torch
import multiprocessing
import gc
import time
import json

login(token="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")

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True
if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.9)
else:
    torch.set_num_threads(os.cpu_count())

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    device=0 if device == "cuda" else -1
)

MAX_LENGTH = 2000
MAX_WORKERS = os.cpu_count()
LOG_FILE = 'errores_procesamiento.json'
BATCH_SIZE = 55

multiprocessing.set_start_method("spawn", force=True)

def process_text_with_ai(texts, qa_pipeline):
    respuestas = []
    for text in texts:
        if not text.strip():
            respuestas.append(None)
            continue
        try:
            text = text[:MAX_LENGTH]
            result = qa_pipeline(question="¿Quién es el autor del libro?", context=text)
            respuestas.append(result.get('answer', None))
        except Exception as e:
            respuestas.append(None)
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)
    return respuestas

def process_pdf(ruta_archivo):
    lector = PdfReader(ruta_archivo)
    num_paginas = min(10, len(lector.pages))
    texto = ''
    for num_pagina in range(num_paginas):
        pagina = lector.pages[num_pagina]
        try:
            if pagina.extract_text():
                texto += pagina.extract_text() + '\n'
        except Exception:
            continue
    return texto

def process_epub(ruta_archivo):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        libro = epub.read_epub(ruta_archivo)
    texto = ''
    conteo = 0
    for item in libro.get_items():
        if item.get_type() == epub.EpubHtml:
            try:
                contenido = item.get_content().decode('utf-8')
                contenido = re.sub(r'<[^>]+>', '', contenido)
                texto += contenido + '\n'
                conteo += 1
                if conteo >= 10:
                    break
            except Exception:
                continue
    return texto

def process_docx(ruta_archivo):
    documento = docx.Document(ruta_archivo)
    texto = ''
    parrafos_por_pagina = 30
    num_parrafos = min(10 * parrafos_por_pagina, len(documento.paragraphs))
    for i in range(num_parrafos):
        texto += documento.paragraphs[i].text + '\n'
    return texto

def process_file(ruta_archivo):
    ext = os.path.splitext(ruta_archivo)[1].lower()
    if ext == '.pdf':
        return process_pdf(ruta_archivo)
    elif ext == '.epub':
        return process_epub(ruta_archivo)
    elif ext == '.docx':
        return process_docx(ruta_archivo)
    else:
        return None

def main():
    carpeta_entrada = 'Libros'
    carpeta_salida = 'Libros_Organizados'
    archivos_error = []
    archivos_no_soportados = []

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    archivos_para_procesar = []
    for root, _, files in os.walk(carpeta_entrada):
        for nombre_archivo in files:
            ruta_archivo = os.path.join(root, nombre_archivo)
            if os.path.isfile(ruta_archivo):
                archivos_para_procesar.append(ruta_archivo)

    textos_para_procesar = []
    rutas_archivos = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, ruta_archivo): ruta_archivo for ruta_archivo in archivos_para_procesar}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extrayendo textos de archivos", unit="archivo"):
            try:
                texto = future.result()
                ruta_archivo = futures[future]
                if texto is None:
                    archivos_no_soportados.append(ruta_archivo)
                else:
                    textos_para_procesar.append(texto)
                    rutas_archivos.append(ruta_archivo)
            except Exception as e:
                archivos_error.append((futures[future], str(e)))

    resultados = []
    for i in tqdm(range(0, len(textos_para_procesar), BATCH_SIZE), desc="Procesando textos con AI", unit="lote"):
        batch_textos = textos_para_procesar[i:i + BATCH_SIZE]
        batch_resultados = process_text_with_ai(batch_textos, qa_pipeline)
        resultados.extend(batch_resultados)

    for idx, autor in enumerate(resultados):
        ruta_archivo = rutas_archivos[idx]
        nombre_archivo = os.path.basename(ruta_archivo)
        try:
            if not autor or autor.strip() == '' or autor.lower() == 'no answer':
                carpeta_autor = os.path.join(carpeta_salida, 'Autor_Desconocido')
            else:
                nombre_autor = re.sub(r'[<>:"/\\|?*]', '', autor)
                carpeta_autor = os.path.join(carpeta_salida, nombre_autor)

            if not os.path.exists(carpeta_autor):
                os.makedirs(carpeta_autor)

            shutil.copy2(ruta_archivo, os.path.join(carpeta_autor, nombre_archivo))
        except Exception as e:
            archivos_error.append((nombre_archivo, str(e)))

    log_data = {
        "archivos_error": archivos_error,
        "archivos_no_soportados": archivos_no_soportados
    }
    with open(LOG_FILE, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

if __name__ == '__main__':
    main()