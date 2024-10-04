import os
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from multiprocessing import Manager

from analysis import extract_authors_batch
from file_reader import process_file
from organizer import organize_file
from utils import log_data, log_error

load_dotenv()

CARPETA_ENTRADA = '/mnt/FASTDATA/LibrosBiblioteca'
CARPETA_SALIDA = 'Libros_Organizados'
LOG_FILE = 'errores_procesamiento.json'
MAX_WORKERS = os.cpu_count()
BATCH_SIZE = 64

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
        futures = {executor.submit(process_file, ruta_archivo, ext): ruta_archivo for ruta_archivo, ext in archivos_para_procesar}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extrayendo textos de archivos", unit="archivo"):
            ruta_archivo = futures[future]
            try:
                result, author = future.result()
                if result is not None:
                    textos_para_procesar.append(result)
                    rutas_archivos.append(ruta_archivo)
                    autores_extraidos.append(author)
                else:
                    log_error(ruta_archivo, "No se pudo extraer texto del archivo.")
            except Exception as e:
                log_error(ruta_archivo, str(e))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in tqdm(range(0, len(textos_para_procesar), BATCH_SIZE), desc="Extrayendo autores", unit="lote"):
            batch_texts = textos_para_procesar[i:i+BATCH_SIZE]
            batch_authors = autores_extraidos[i:i+BATCH_SIZE]
            batch_rutas = rutas_archivos[i:i+BATCH_SIZE]
            autores_extraidos[i:i+BATCH_SIZE] = list(executor.map(
                lambda x: extract_authors_batch(*x), 
                zip(batch_texts, batch_authors, batch_rutas)
            ))

    manager = Manager()
    known_authors = manager.list()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(organize_file, ruta_archivo, author, known_authors) for ruta_archivo, author in zip(rutas_archivos, autores_extraidos)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Organizando archivos por autor", unit="archivo"):
            future.result()

    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        json.dump(log_data, log_file, indent=4, ensure_ascii=False)

    print("Terminé")

if __name__ == '__main__':
    main()
