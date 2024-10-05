import torch
from transformers import pipeline
from utils import log_error, clean_input_text
from file_types import RESPUESTA_IA_NO_ENCONTRADA, QUESTIONS_AUTHOR_VARIATIONS

QUESTION_AUTHOR = "¿Quién es el autor del libro?"
MAX_CHARACTERS = 15000

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.9)
else:
    torch.set_num_threads(os.cpu_count())

tqa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    device=0 if device == "cuda" else -1,
    clean_up_tokenization_spaces=False
)

ner_pipeline = pipeline(
    "ner",
    model="dccuchile/bert-base-spanish-wwm-cased-finetuned-ner",
    device=0 if device == "cuda" else -1
)

def extract_author_using_ner(text):
    ner_results = ner_pipeline(text)
    author_candidates = [entity['word'] for entity in ner_results if 'PER' in entity['entity']]
    if author_candidates:
        return ' '.join(author_candidates)
    return None

def extract_authors_batch(text, author, ruta_archivo, metadata, batch_size):
    try:
        if author:
            log_error(ruta_archivo, "Author metadata already provided.")
            return author

        text = clean_input_text(text)
        if len(text) < 100:
            log_error(ruta_archivo, "Text too short for meaningful analysis.")
            return None

        context_with_metadata = f"Título: {metadata.get('title', '')}, Archivo: {metadata.get('filename', '')}\n\n{text}"

        for question in QUESTIONS_AUTHOR_VARIATIONS:
            qa_inputs = {'context': context_with_metadata, 'question': question}

            try:
                answer = tqa_pipeline(qa_inputs).get('answer', None)
                if answer and answer.lower() not in RESPUESTA_IA_NO_ENCONTRADA:
                    log_error(ruta_archivo, f"Answer found with question '{question}'")
                    return answer
            except Exception as e:
                log_error(ruta_archivo, f"Error processing QA for question '{question}': {e}")

        author_from_ner = extract_author_using_ner(text)
        if author_from_ner:
            log_error(ruta_archivo, "Author found using NER.")
            return author_from_ner

        log_error(ruta_archivo, "No se pudo determinar el autor.")
        return None
    except Exception as e:
        log_error(ruta_archivo, f"Exception in extract_authors_batch: {e}")
        return None
