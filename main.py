import os
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue
from threading import Thread
from dotenv import load_dotenv
from analysis import extract_authors_batch
from file_reader import process_file
from organizer import organize_file
from utils import log_data, log_error, cargar_archivos, contar_archivos

load_dotenv()

CARPETA_ENTRADA = 'Libros'
CARPETA_SALIDA = 'Libros_Organizados'
LOG_FILE = 'errores_procesamiento.json'
MAX_WORKERS = os.cpu_count()
BATCH_SIZE = 64

def procesar_archivos(cola_archivos, cola_analisis, total_archivos):
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=total_archivos, desc="Extrayendo texto de archivos", unit="archivo") as pbar:
            while True:
                batch_archivos = cola_archivos.get()
                if batch_archivos == "FIN":
                    cola_analisis.put("FIN")
                    log_error("procesar_archivos", "Proceso de extracción de textos finalizado.")
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
                    pbar.update(1)

                if len(textos_para_procesar) > 0:
                    cola_analisis.put((textos_para_procesar, autores_extraidos, rutas_archivos))

def analizar_autores(cola_analisis, cola_organizacion, total_archivos):
    with tqdm(total=total_archivos, desc="Analizando autores", unit="archivo") as pbar:
        while True:
            batch_data = cola_analisis.get()
            if batch_data == "FIN":
                cola_organizacion.put("FIN")
                log_error("analizar_autores", "Proceso de análisis de autores finalizado.")
                break

            textos_para_procesar, autores_extraidos, rutas_archivos = batch_data

            try:
                batch_autores = list(map(
                    lambda x: extract_authors_batch(*x, BATCH_SIZE), 
                    zip(textos_para_procesar, autores_extraidos, rutas_archivos)
                ))

                cola_organizacion.put((rutas_archivos, batch_autores))
            except Exception as e:
                log_error("analizar_autores", str(e))

            pbar.update(len(rutas_archivos))

def organizar_archivos(cola_organizacion, known_authors, total_archivos):
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=total_archivos, desc="Organizando archivos", unit="archivo") as pbar:
            while True:
                batch_data = cola_organizacion.get()
                if batch_data == "FIN":
                    log_error("organizar_archivos", "Proceso de organización finalizado.")
                    break

                rutas_archivos, batch_autores = batch_data

                futuros = [executor.submit(organize_file, ruta_archivo, author, known_authors)
                           for ruta_archivo, author in zip(rutas_archivos, batch_autores)]

                for futuro in as_completed(futuros):
                    try:
                        futuro.result()
                    except Exception as e:
                        log_error(rutas_archivos, str(e))
                    pbar.update(1)

def main():
    try:
        if not os.path.exists(CARPETA_SALIDA):
            os.makedirs(CARPETA_SALIDA)

        cola_archivos = Queue()
        cola_analisis = Queue()
        cola_organizacion = Queue()

        manager = Manager()
        known_authors = manager.list()

        thread_cargar = Thread(target=cargar_archivos, args=(cola_archivos, CARPETA_ENTRADA, BATCH_SIZE))
        thread_cargar.start()

        total_archivos = contar_archivos(CARPETA_ENTRADA)

        thread_procesar = Thread(target=procesar_archivos, args=(cola_archivos, cola_analisis, total_archivos))
        thread_analizar = Thread(target=analizar_autores, args=(cola_analisis, cola_organizacion, total_archivos))
        thread_organizar = Thread(target=organizar_archivos, args=(cola_organizacion, known_authors, total_archivos))

        thread_procesar.start()
        thread_analizar.start()
        thread_organizar.start()

        thread_cargar.join()

        cola_archivos.put("FIN")
        thread_procesar.join()
        cola_analisis.put("FIN")
        thread_analizar.join()
        cola_organizacion.put("FIN")
        thread_organizar.join()

        with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
            json.dump(log_data, log_file, indent=4, ensure_ascii=False)

        log_error("main", "Procesamiento terminado.")
        print("Todo el proceso ha terminado correctamente.")
    except Exception as e:
        log_error("main", str(e))

if __name__ == '__main__':
    main()
