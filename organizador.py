import os
import shutil
import re
import json
import time
import warnings
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login
import torch
from PyPDF2 import PdfReader
from ebooklib import epub
import docx

# Login to Hugging Face (replace with your own token or use environment variable)
login(token="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")

# Determine the device to use (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Optimize PyTorch settings
torch.backends.cudnn.benchmark = True
if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.9)
else:
    torch.set_num_threads(os.cpu_count())

# Initialize the question-answering pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    device=0 if device == "cuda" else -1
)

# Constants
MAX_LENGTH = 2000  # Max length of context text
MAX_WORKERS = os.cpu_count()
LOG_FILE = 'errores_procesamiento.json'
BATCH_SIZE = 16  # Adjust batch size to fit into GPU memory

def process_text_with_ai(texts, qa_pipeline):
    """
    Processes a list of texts with the QA pipeline to extract authors.
    """
    # Prepare inputs for the pipeline
    inputs = []
    for text in texts:
        if text.strip():
            inputs.append({
                'question': '¿Quién es el autor del libro?',
                'context': text[:MAX_LENGTH]
            })
        else:
            inputs.append(None)

    # Process inputs in batches
    resultados = []
    for i in tqdm(range(0, len(inputs), BATCH_SIZE), desc="Procesando textos con IA", unit="batch"):
        batch_inputs = [inp for inp in inputs[i:i+BATCH_SIZE] if inp]
        if not batch_inputs:
            resultados.extend([None] * (len(inputs[i:i+BATCH_SIZE])))
            continue
        try:
            batch_outputs = qa_pipeline(batch_inputs, batch_size=BATCH_SIZE)
            batch_answers = [output.get('answer', None) for output in batch_outputs]
            # Handle the cases where inputs were None
            idx = 0
            for inp in inputs[i:i+BATCH_SIZE]:
                if inp:
                    resultados.append(batch_answers[idx])
                    idx += 1
                else:
                    resultados.append(None)
        except Exception as e:
            # In case of error, append None for each input in the batch
            resultados.extend([None] * len(inputs[i:i+BATCH_SIZE]))
            print(f"Error processing batch: {e}")
        finally:
            # Clear cache to prevent memory leaks
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    return resultados

def process_pdf(ruta_archivo):
    """
    Extracts text from the first 10 pages of a PDF file.
    """
    try:
        lector = PdfReader(ruta_archivo)
        num_paginas = min(10, len(lector.pages))
        texto = ''
        for num_pagina in range(num_paginas):
            pagina = lector.pages[num_pagina]
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto += texto_pagina + '\n'
        return texto
    except Exception as e:
        print(f"Error processing PDF {ruta_archivo}: {e}")
        return None

def process_epub(ruta_archivo):
    """
    Extracts text from the first 10 items of an EPUB file.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            libro = epub.read_epub(ruta_archivo)
        texto = ''
        conteo = 0
        for item in libro.get_items():
            if item.get_type() == epub.EpubHtml:
                contenido = item.get_content().decode('utf-8')
                contenido = re.sub(r'<[^>]+>', '', contenido)
                texto += contenido + '\n'
                conteo += 1
                if conteo >= 10:
                    break
        return texto
    except Exception as e:
        print(f"Error processing EPUB {ruta_archivo}: {e}")
        return None

def process_docx(ruta_archivo):
    """
    Extracts text from the first 10 pages (estimated) of a DOCX file.
    """
    try:
        documento = docx.Document(ruta_archivo)
        texto = ''
        parrafos_por_pagina = 30  # Estimated number of paragraphs per page
        num_parrafos = min(10 * parrafos_por_pagina, len(documento.paragraphs))
        for i in range(num_parrafos):
            texto += documento.paragraphs[i].text + '\n'
        return texto
    except Exception as e:
        print(f"Error processing DOCX {ruta_archivo}: {e}")
        return None

def process_file(ruta_archivo):
    """
    Processes a file based on its extension and extracts text.
    """
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

    # Collect files to process
    archivos_para_procesar = []
    for root, _, files in os.walk(carpeta_entrada):
        for nombre_archivo in files:
            ruta_archivo = os.path.join(root, nombre_archivo)
            if os.path.isfile(ruta_archivo):
                archivos_para_procesar.append(ruta_archivo)

    # Extract texts from files using ThreadPoolExecutor
    textos_para_procesar = []
    rutas_archivos = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, ruta_archivo): ruta_archivo for ruta_archivo in archivos_para_procesar}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extrayendo textos de archivos", unit="archivo"):
            ruta_archivo = futures[future]
            try:
                texto = future.result()
                if texto is None:
                    archivos_no_soportados.append(ruta_archivo)
                else:
                    textos_para_procesar.append(texto)
                    rutas_archivos.append(ruta_archivo)
            except Exception as e:
                archivos_error.append((ruta_archivo, str(e)))

    # Process texts with AI to extract authors
    resultados = process_text_with_ai(textos_para_procesar, qa_pipeline)

    # Organize files based on authors
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
            archivos_error.append((ruta_archivo, str(e)))

    # Write error logs to file
    log_data = {
        "archivos_error": archivos_error,
        "archivos_no_soportados": archivos_no_soportados
    }
    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        json.dump(log_data, log_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
