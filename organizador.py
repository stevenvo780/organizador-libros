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

# Autenticación en Hugging Face
login(token="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")

# Configuración de dispositivos para PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ajustes para maximizar el uso de GPU con una RTX 2060 y un i5 10400
torch.backends.cudnn.benchmark = True  # Activar benchmark de CuDNN para mejorar el rendimiento en operaciones repetitivas
torch.set_num_threads(6)  # Ajustar el número de hilos a 6 para un mejor uso de la CPU de 6 núcleos

# Cargar el modelo de QA una vez
qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    device=0 if device == "cuda" else -1
)

MAX_LENGTH = 1000  # Límite de caracteres para el contexto
MAX_WORKERS = 2  # Límite de hilos para el ThreadPoolExecutor
LOG_FILE = 'errores_procesamiento.json'  # Archivo para registrar errores

# Cambiar el método de inicio de los procesos a 'spawn' para evitar errores con CUDA
multiprocessing.set_start_method("spawn", force=True)

def process_text_with_ai(text, qa_pipeline):
    if not text.strip():
        return None  # Si el contexto está vacío, devolver None directamente
    try:
        text = text[:MAX_LENGTH]
        result = qa_pipeline(question="¿Quién es el autor del libro?", context=text)
        return result.get('answer', None)
    except Exception as e:
        return None  # Devolver None y registrar el error en el log general
    finally:
        # Liberar memoria de GPU
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)  # Esperar un segundo para permitir la liberación de memoria


def process_pdf(ruta_archivo, qa_pipeline):
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
    return process_text_with_ai(texto, qa_pipeline)


def process_epub(ruta_archivo, qa_pipeline):
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
    return process_text_with_ai(texto, qa_pipeline)


def process_docx(ruta_archivo, qa_pipeline):
    documento = docx.Document(ruta_archivo)
    texto = ''
    parrafos_por_pagina = 30
    num_parrafos = min(10 * parrafos_por_pagina, len(documento.paragraphs))
    for i in range(num_parrafos):
        texto += documento.paragraphs[i].text + '\n'
    return process_text_with_ai(texto, qa_pipeline)


def process_file(ruta_archivo, carpeta_salida, qa_pipeline):
    archivos_error = []
    archivos_no_soportados = []
    nombre_archivo = os.path.basename(ruta_archivo)
    ext = os.path.splitext(nombre_archivo)[1].lower()
    try:
        if ext == '.pdf':
            autor = process_pdf(ruta_archivo, qa_pipeline)
        elif ext == '.epub':
            autor = process_epub(ruta_archivo, qa_pipeline)
        elif ext == '.docx':
            autor = process_docx(ruta_archivo, qa_pipeline)
        else:
            archivos_no_soportados.append(nombre_archivo)
            return archivos_error, archivos_no_soportados

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

    return archivos_error, archivos_no_soportados


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

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, ruta_archivo, carpeta_salida, qa_pipeline): ruta_archivo for ruta_archivo in archivos_para_procesar}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando archivos"):
            errores, no_soportados = future.result()
            archivos_error.extend(errores)
            archivos_no_soportados.extend(no_soportados)

    # Guardar errores en un archivo JSON
    log_data = {
        "archivos_error": archivos_error,
        "archivos_no_soportados": archivos_no_soportados
    }
    with open(LOG_FILE, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

if __name__ == '__main__':
    main()