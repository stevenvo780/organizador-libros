import torch
from transformers import pipeline
from utils import log_error

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

ner_pipeline = pipeline("ner", model="dccuchile/bert-base-spanish-wwm-cased-finetuned-ner", device=0 if device == "cuda" else -1)

def extract_author_using_ner(text):
    ner_results = ner_pipeline(text)
    author_candidates = [entity['word'] for entity in ner_results if entity['entity'] == 'B-PER']
    if author_candidates:
        return ' '.join(author_candidates)
    return None

def extract_authors_batch(text, author, ruta_archivo, batch_size):
    if author:
        return author
    
    qa_inputs = {'context': text[:MAX_CHARACTERS].strip(), 'question': QUESTION_AUTHOR}

    try:
        answer = tqa_pipeline(qa_inputs, batch_size=batch_size)[0].get('answer', None)
        if answer and answer.lower() not in ['no sé', 'no answer']:
            return answer
    except Exception as e:
        log_error(ruta_archivo, f"Error processing QA: {e}")

    author_from_ner = extract_author_using_ner(text)
    if author_from_ner:
        return author_from_ner
    
    log_error(ruta_archivo, "No se pudo determinar el autor.")
    return None
