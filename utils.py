import re
import unicodedata
from difflib import SequenceMatcher
import os
from file_types import FORMATOS_ARCHIVOS

MAX_CHARACTERS = 15000  # Added MAX_CHARACTERS definition

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
    return ' '.join(name.split())

def get_best_matching_author(name, known_authors):
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
        known_authors.append(name)
        return name

def cargar_archivos(cola_archivos, CARPETA_ENTRADA, BATCH_SIZE):
    archivos_para_procesar = []
    for root, _, files in os.walk(CARPETA_ENTRADA):
        for nombre_archivo in files:
            ruta_archivo = os.path.join(root, nombre_archivo)
            if os.path.isfile(ruta_archivo):
                ext = os.path.splitext(ruta_archivo)[1].lower()
                if any(ext in formatos for formatos in FORMATOS_ARCHIVOS.values()):
                    archivos_para_procesar.append((ruta_archivo, ext))
                else:
                    log_data["archivos_no_soportados"].append(ruta_archivo)

    for i in range(0, len(archivos_para_procesar), BATCH_SIZE):
        batch_archivos = archivos_para_procesar[i:i+BATCH_SIZE]
        cola_archivos.put(batch_archivos)

def contar_archivos(CARPETA_ENTRADA):
    total_archivos = 0
    for root, _, files in os.walk(CARPETA_ENTRADA):
        for nombre_archivo in files:
            ruta_archivo = os.path.join(root, nombre_archivo)
            if os.path.isfile(ruta_archivo):
                ext = os.path.splitext(ruta_archivo)[1].lower()
                if any(ext in formatos for formatos in FORMATOS_ARCHIVOS.values()):
                    total_archivos += 1
    return total_archivos

def clean_input_text(text):
    if not text:
        return ''
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text[:MAX_CHARACTERS]
