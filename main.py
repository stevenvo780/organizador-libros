import os
import json
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

from analysis import extract_authors_batch
from file_reader import process_file
from organizer import organize_file, log_data, log_error

load_dotenv()

CARPETA_ENTRADA = '/mnt/FASTDATA/LibrosBiblioteca'
CARPETA_SALIDA = 'Libros_Organizados'
LOG_FILE = 'errores_procesamiento.json'
MAX_WORKERS = os.cpu_count()

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

    autores_extraidos = extract_authors_batch(textos_para_procesar, autores_extraidos, rutas_archivos)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(organize_file, (idx, author, rutas_archivos)) for idx, author in enumerate(autores_extraidos)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Organizando archivos por autor", unit="archivo"):
            future.result()

    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        json.dump(log_data, log_file, indent=4, ensure_ascii=False)

    print("Terminé")

if __name__ == '__main__':
    main()
