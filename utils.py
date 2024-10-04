import re
import unicodedata

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
