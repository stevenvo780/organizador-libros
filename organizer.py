import os
import re
import shutil
from utils import log_error

CARPETA_SALIDA = 'Libros_Organizados'

def organize_file(ruta_archivo, nombre_autor):
    nombre_archivo = os.path.basename(ruta_archivo)
    try:
        nombre_autor = re.sub(r'[<>:"/\\|?*]', '', nombre_autor)
        carpeta_autor = os.path.join(CARPETA_SALIDA, nombre_autor)
        if not os.path.exists(carpeta_autor):
            os.makedirs(carpeta_autor)
        destino = os.path.join(carpeta_autor, nombre_archivo)
        if not os.path.exists(destino):
            shutil.copy2(ruta_archivo, destino)
    except Exception as e:
        log_error(ruta_archivo, str(e))
