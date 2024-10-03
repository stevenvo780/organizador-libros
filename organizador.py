import os
import shutil
import re
from PyPDF2 import PdfReader
from ebooklib import epub
import docx
from transformers import pipeline

def process_text_with_ai(text):
    """
    Utiliza un modelo de lenguaje basado en IA para extraer el nombre del autor del texto.
    """
    # Definir la pregunta para el modelo
    question = "¿Quién es el autor del libro?"
    # Crear una instancia del pipeline de preguntas y respuestas
    qa_pipeline = pipeline("question-answering", model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")

    # Preparar el contexto y la pregunta para el modelo
    result = qa_pipeline(question=question, context=text)

    # Obtener la respuesta
    autor = result.get('answer', None)
    return autor

def process_pdf(ruta_archivo):
    """
    Función para extraer texto de las primeras 10 páginas de un archivo PDF.
    """
    try:
        lector = PdfReader(ruta_archivo)
        num_paginas = min(10, len(lector.pages))
        texto = ''
        for num_pagina in range(num_paginas):
            pagina = lector.pages[num_pagina]
            texto += pagina.extract_text() + '\n'
        autor = process_text_with_ai(texto)
        return autor
    except Exception as e:
        raise e

def process_epub(ruta_archivo):
    """
    Función para extraer texto de los primeros 10 ítems (capítulos/páginas) de un archivo EPUB.
    """
    try:
        libro = epub.read_epub(ruta_archivo)
        texto = ''
        conteo = 0
        for item in libro.get_items():
            if item.get_type() == epub.EpubHtml:
                contenido = item.get_content().decode('utf-8')
                # Eliminar etiquetas HTML
                contenido = re.sub(r'<[^>]+>', '', contenido)
                texto += contenido + '\n'
                conteo += 1
                if conteo >= 10:
                    break
        autor = process_text_with_ai(texto)
        return autor
    except Exception as e:
        raise e

def process_docx(ruta_archivo):
    """
    Función para extraer texto de las primeras 10 páginas de un archivo DOCX.
    """
    try:
        documento = docx.Document(ruta_archivo)
        texto = ''
        parrafos_por_pagina = 30  # Aproximado de párrafos por página
        num_parrafos = min(10 * parrafos_por_pagina, len(documento.paragraphs))
        for i in range(num_parrafos):
            texto += documento.paragraphs[i].text + '\n'
        autor = process_text_with_ai(texto)
        return autor
    except Exception as e:
        raise e

def main():
    # Definir la carpeta raíz donde se encuentran los libros
    carpeta_entrada = 'Libros'  # Cambia esto por la carpeta que contiene tus libros
    carpeta_salida = 'Libros_Organizados'  # Carpeta donde se guardarán los libros organizados

    # Listas para rastrear errores y formatos no soportados
    archivos_error = []
    archivos_no_soportados = []

    # Crear la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Procesar cada archivo en la carpeta de entrada
    for nombre_archivo in os.listdir(carpeta_entrada):
        ruta_archivo = os.path.join(carpeta_entrada, nombre_archivo)

        # Verificar si es un archivo
        if os.path.isfile(ruta_archivo):
            # Obtener la extensión del archivo
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
                    continue

                # Si no se encontró el autor, mover a 'Autor_Desconocido'
                if not autor or autor.strip() == '':
                    carpeta_autor = os.path.join(carpeta_salida, 'Autor_Desconocido')
                else:
                    # Limpiar el nombre del autor para crear un nombre de carpeta válido
                    nombre_autor = re.sub(r'[<>:"/\\|?*]', '', autor)
                    carpeta_autor = os.path.join(carpeta_salida, nombre_autor)

                # Crear la carpeta del autor si no existe
                if not os.path.exists(carpeta_autor):
                    os.makedirs(carpeta_autor)

                # Mover el archivo a la carpeta del autor
                shutil.move(ruta_archivo, os.path.join(carpeta_autor, nombre_archivo))

            except Exception as e:
                archivos_error.append((nombre_archivo, str(e)))
                continue

    # Reportar errores y archivos no soportados
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
