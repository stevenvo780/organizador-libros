import os
import shutil
import re
from PyPDF2 import PdfReader
from ebooklib import epub
import docx
from transformers import pipeline
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch
from huggingface_hub import login

# Autenticación en Hugging Face
login(token="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")

def process_text_with_ai(text):
    try:
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0, progress_bar=True)
        result = qa_pipeline(question="¿Quién es el autor del libro?", context=text)
        return result.get('answer', None)
    except Exception as e:
        return None

def process_pdf(ruta_archivo):
    lector = PdfReader(ruta_archivo)
    num_paginas = min(10, len(lector.pages))
    texto = ''
    for num_pagina in range(num_paginas):
        pagina = lector.pages[num_pagina]
        try:
            if pagina.extract_text():
                texto += pagina.extract_text() + '\n'
        except Exception as e:
            continue
    return process_text_with_ai(texto)

def process_epub(ruta_archivo):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        libro = epub.read_epub(ruta_archivo)
    texto = ''
    conteo = 0
    for item in libro.get_items():
        if item.get_type() == epub.EpubHtml:
            try:
                contenido = item.get_content().decode('utf-8')
                contenido = re.sub(r'<[^>]+>', '', contenido)
                texto += contenido + '\n'
                conteo += 1
                if conteo >= 10:
                    break
            except Exception as e:
                continue
    return process_text_with_ai(texto)

def process_docx(ruta_archivo):
    documento = docx.Document(ruta_archivo)
    texto = ''
    parrafos_por_pagina = 30
    num_parrafos = min(10 * parrafos_por_pagina, len(documento.paragraphs))
    for i in range(num_parrafos):
        texto += documento.paragraphs[i].text + '\n'
    return process_text_with_ai(texto)

def process_file(ruta_archivo, carpeta_salida):
    archivos_error = []
    archivos_no_soportados = []
    nombre_archivo = os.path.basename(ruta_archivo)
    ext = os.path.splitext(nombre_archivo)[1].lower()
    try:
        if ext == '.pdf':
            autor = process_pdf(ruta_archivo)
        elif ext == '.epub':
            autor = process_epub(ruta_archivo)
        elif ext == '.docx':
            autor = process_docx(ruta_archivo)
        else:
            archivos_no_soportados.append(nombre_archivo)
            return archivos_error, archivos_no_soportados

        if not autor or autor.strip() == '' or autor.lower() == 'no answer':
            carpeta_autor = os.path.join(carpeta_salida, 'Autor_Desconocido')
        else:
            nombre_autor = re.sub(r'[<>:"/\\|?*]', '', autor)
            carpeta_autor = os.path.join(carpeta_salida, nombre_autor)

        if not os.path.exists(carpeta_autor):
            os.makedirs(carpeta_autor)

        shutil.move(ruta_archivo, os.path.join(carpeta_autor, nombre_archivo))

    except Exception as e:
        archivos_error.append((nombre_archivo, str(e)))

    return archivos_error, archivos_no_soportados

def main():
    carpeta_entrada = 'Libros'
    carpeta_salida = 'Libros_Organizados'
    archivos_error = []
    archivos_no_soportados = []

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    archivos_para_procesar = []
    for root, _, files in os.walk(carpeta_entrada):
        for nombre_archivo in files:
            ruta_archivo = os.path.join(root, nombre_archivo)
            if os.path.isfile(ruta_archivo):
                archivos_para_procesar.append(ruta_archivo)

    with ProcessPoolExecutor() as executor:
        resultados = list(tqdm(executor.map(process_file, archivos_para_procesar, [carpeta_salida] * len(archivos_para_procesar)), total=len(archivos_para_procesar), desc="Procesando archivos"))

    for errores, no_soportados in resultados:
        archivos_error.extend(errores)
        archivos_no_soportados.extend(no_soportados)

    if archivos_error:
        print("Ocurrieron errores con los siguientes archivos:")
        for fname, error in archivos_error:
            print(f"{fname}: {error}")

    if archivos_no_soportados:
        print("Los siguientes archivos tienen formatos no soportados:")
        for fname in archivos_no_soportados:
            print(fname)

if __name__ == '__main__':
    main()