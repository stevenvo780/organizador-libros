import os
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue
from threading import Thread
from dotenv import load_dotenv
from analysis import extract_authors_batch  # Importamos desde analysis
from file_reader import process_file
from organizer import organize_file
from utils import log_data, log_error

load_dotenv()

CARPETA_ENTRADA = '/mnt/FASTDATA/LibrosBiblioteca'
CARPETA_SALIDA = 'Libros_Organizados'
LOG_FILE = 'errores_procesamiento.json'
MAX_WORKERS = os.cpu_count()
BATCH_SIZE = 64  # Aquí definimos el tamaño de los lotes

def cargar_archivos(cola_archivos):
    archivos_para_procesar = []
    for root, _, files in os.walk(CARPETA_ENTRADA):
        for nombre_archivo in files:
            ruta_archivo = os.path.join(root, nombre_archivo)
            if os.path.isfile(ruta_archivo):
                ext = os.path.splitext(ruta_archivo)[1].lower()
                if ext in ['.pdf', '.epub', '.docx', '.doc', '.rtf']:
                    archivos_para_procesar.append((ruta_archivo, ext))
                else:
                    log_data["archivos_no_soportados"].append(ruta_archivo)
    
    for i in range(0, len(archivos_para_procesar), BATCH_SIZE):
        batch_archivos = archivos_para_procesar[i:i+BATCH_SIZE]
        cola_archivos.put(batch_archivos)

    return len(archivos_para_procesar)

def procesar_archivos(cola_archivos, cola_analisis, total_archivos):
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        processed_files = 0
        with tqdm(total=total_archivos, desc="Extrayendo texto de archivos", unit="archivo") as pbar:
            while True:
                batch_archivos = cola_archivos.get()
                if batch_archivos == "FIN":
                    cola_analisis.put("FIN")
                    break

                futuros = {executor.submit(process_file, ruta_archivo, ext): ruta_archivo for ruta_archivo, ext in batch_archivos}
                textos_para_procesar = []
                rutas_archivos = []
                autores_extraidos = []

                for futuro in as_completed(futuros):
                    ruta_archivo = futuros[futuro]
                    try:
                        result, author = futuro.result()
                        if result is not None:
                            textos_para_procesar.append(result)
                            rutas_archivos.append(ruta_archivo)
                            autores_extraidos.append(author)
                        else:
                            log_error(ruta_archivo, "No se pudo extraer texto del archivo.")
                    except Exception as e:
                        log_error(ruta_archivo, str(e))
                    processed_files += 1
                    pbar.update(1)

                cola_analisis.put((textos_para_procesar, autores_extraidos, rutas_archivos))

def analizar_autores(cola_analisis, cola_organizacion, total_archivos, batch_size):
    processed_authors = 0
    with tqdm(total=total_archivos, desc="Analizando autores", unit="archivo") as pbar:
        while True:
            batch_data = cola_analisis.get()
            if batch_data == "FIN":
                cola_organizacion.put("FIN")
                break

            textos_para_procesar, autores_extraidos, rutas_archivos = batch_data

            batch_autores = list(map(
                lambda x: extract_authors_batch(*x, batch_size), 
                zip(textos_para_procesar, autores_extraidos, rutas_archivos)
            ))

            cola_organizacion.put((rutas_archivos, batch_autores))

            processed_authors += len(rutas_archivos)
            pbar.update(len(rutas_archivos))

def organizar_archivos(cola_organizacion, known_authors, total_archivos):
    organized_files = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=total_archivos, desc="Organizando archivos", unit="archivo") as pbar:
            while True:
                batch_data = cola_organizacion.get()
                if batch_data == "FIN":
                    break

                rutas_archivos, batch_autores = batch_data
                futuros = [executor.submit(organize_file, ruta_archivo, author, known_authors)
                           for ruta_archivo, author in zip(rutas_archivos, batch_autores)]

                for futuro in as_completed(futuros):
                    futuro.result()
                    organized_files += 1
                    pbar.update(1)

def main():
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)

    cola_archivos = Queue()
    cola_analisis = Queue()
    cola_organizacion = Queue()

    manager = Manager()
    known_authors = manager.list()

    thread_cargar = Thread(target=cargar_archivos, args=(cola_archivos,))
    thread_cargar.start()
    total_archivos = cargar_archivos(cola_archivos)

    thread_procesar = Thread(target=procesar_archivos, args=(cola_archivos, cola_analisis, total_archivos))
    thread_analizar = Thread(target=analizar_autores, args=(cola_analisis, cola_organizacion, total_archivos, BATCH_SIZE))  # Pasamos BATCH_SIZE aquí
    thread_organizar = Thread(target=organizar_archivos, args=(cola_organizacion, known_authors, total_archivos))

    thread_procesar.start()
    thread_analizar.start()
    thread_organizar.start()

    thread_cargar.join()

    cola_archivos.put("FIN")

    thread_procesar.join()
    thread_analizar.join()
    thread_organizar.join()

    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        json.dump(log_data, log_file, indent=4, ensure_ascii=False)

    print("Terminé")

if __name__ == '__main__':
    main()
