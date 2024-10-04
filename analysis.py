import torch
from transformers import pipeline
from difflib import SequenceMatcher
from utils import log_error

QUESTION_AUTHOR = "¿Quién es el autor del libro?"
MAX_CHARACTERS = 15000
BATCH_SIZE = 64

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

ner_pipeline = pipeline("ner", model="dccuchile/bert-base-spanish-wwm-cased-finetuned-ner", device=0 if device == "cuda" else -1)

def extract_author_using_ner(text):
    ner_results = ner_pipeline(text)
    author_candidates = [entity['word'] for entity in ner_results if entity['entity'] == 'B-PER']
    if author_candidates:
        return ' '.join(author_candidates)
    return None

def extract_authors_batch(texts, autores_extraidos, rutas_archivos):
    indices_sin_autor = [i for i, autor in enumerate(autores_extraidos) if not autor]
    if indices_sin_autor:
        for batch_start in range(0, len(indices_sin_autor), BATCH_SIZE):
            batch_indices = indices_sin_autor[batch_start:batch_start + BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_indices]

            qa_inputs = []
            valid_indices = []
            for idx, text in zip(batch_indices, batch_texts):
                context = text[:MAX_CHARACTERS].strip()
                if not context:
                    log_error(rutas_archivos[idx], "Contexto vacío, no se puede extraer autor")
                    continue
                qa_inputs.append({'context': context, 'question': QUESTION_AUTHOR})
                valid_indices.append(idx)

            try:
                outputs = tqa_pipeline(qa_inputs, batch_size=BATCH_SIZE)
                for idx_output, output in zip(valid_indices, outputs):
                    answer = output.get('answer', None)
                    autores_extraidos[idx_output] = answer if answer else extract_author_using_ner(texts[idx_output])
            except Exception as e:
                for idx_error in valid_indices:
                    log_error(rutas_archivos[idx_error], f"Error processing QA: {e}")
                    autores_extraidos[idx_error] = None
    return autores_extraidos
