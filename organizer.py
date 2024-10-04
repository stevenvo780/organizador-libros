import os
import re
import shutil
import unicodedata

log_data = {
    "archivos_error": [],
    "archivos_no_soportados": []
}

CARPETA_SALIDA = 'Libros_Organizados'

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
        known_authors.add(name)
        return name

def organize_file(args):
    idx, author, rutas_archivos = args
    ruta_archivo = rutas_archivos[idx]
    nombre_archivo = os.path.basename(ruta_archivo)
    known_authors = set()
    try:
        if not author or author.strip() == '' or author.lower() == 'no answer':
            nombre_autor = 'Autor Desconocido'
        else:
            nombre_autor = normalize_author_name(author)
            nombre_autor = get_best_matching_author(nombre_autor, known_authors)
            nombre_autor = re.sub(r'[<>:"/\\|?*]', '', nombre_autor)

        carpeta_autor = os.path.join(CARPETA_SALIDA, nombre_autor)

        if not os.path.exists(carpeta_autor):
            os.makedirs(carpeta_autor)

        destino = os.path.join(carpeta_autor, nombre_archivo)
        if not os.path.exists(destino):
            shutil.copy2(ruta_archivo, destino)
    except Exception as e:
        log_error(ruta_archivo, str(e))
