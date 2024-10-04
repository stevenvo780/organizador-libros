import os
import shutil
import re
import json
import warnings
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login
import torch
from PyPDF2 import PdfReader
from ebooklib import epub
import docx
from datasets import Dataset

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

def clean_text(text):
    """
    Cleans text to remove any invalid characters that may cause UnicodeEncodeError.
    """
    # Replace surrogate characters with a space
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return text

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
        return clean_text(texto)
    except Exception as e:
        return f"Error processing PDF {ruta_archivo}: {e}"

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
        return clean_text(texto)
    except Exception as e:
        return f"Error processing EPUB {ruta_archivo}: {e}"

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
        return clean_text(texto)
    except Exception as e:
        return f"Error processing DOCX {ruta_archivo}: {e}"

def process_file(args):
    """
    Processes a file based on its extension and extracts text.
    Returns a tuple (ruta_archivo, texto extraído o mensaje de error)
    """
    ruta_archivo, ext = args
    if ext == '.pdf':
        result = process_pdf(ruta_archivo)
    elif ext == '.epub':
        result = process_epub(ruta_archivo)
    elif ext == '.docx':
        result = process_docx(ruta_archivo)
    else:
        result = None  # Unsupported file type
    return ruta_archivo, result

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
                ext = os.path.splitext(ruta_archivo)[1].lower()
                if ext in ['.pdf', '.epub', '.docx']:
                    archivos_para_procesar.append((ruta_archivo, ext))
                else:
                    archivos_no_soportados.append(ruta_archivo)

    # Extract texts from files using ProcessPoolExecutor
    textos_para_procesar = []
    rutas_archivos = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, args): args[0] for args in archivos_para_procesar}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extrayendo textos de archivos", unit="archivo"):
            ruta_archivo = futures[future]
            try:
                ruta_archivo, result = future.result()
                if result is None or result.startswith("Error processing"):
                    archivos_error.append((ruta_archivo, result))
                else:
                    textos_para_procesar.append(result)
                    rutas_archivos.append(ruta_archivo)
            except Exception as e:
                archivos_error.append((ruta_archivo, str(e)))

    # Prepare dataset for batch processing with AI
    cleaned_texts = [clean_text(text) for text in textos_para_procesar]
    dataset = Dataset.from_dict({
        'context': cleaned_texts,
        'question': ['¿Quién es el autor del libro?'] * len(cleaned_texts)
    })

    # Process texts with AI using dataset and batch mode
    print("Procesando textos con IA...")
    resultados = []
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Procesando textos con IA", unit="batch"):
        batch = dataset.select(range(i, min(i + BATCH_SIZE, len(dataset))))
        try:
            batch_results = qa_pipeline(batch.to_pandas().to_dict(orient="records"), batch_size=BATCH_SIZE)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
            batch_answers = [output.get('answer', None) for output in batch_results]
            resultados.extend(batch_answers)
        except Exception as e:
            resultados.extend([None] * len(batch))
            print(f"Error processing batch: {e}")
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

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
