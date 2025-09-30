import numpy as np
import torch
import torch.nn as nn
import hashlib
import joblib
from collections import Counter
import gradio as gr

# --- utils (from the notebook) ---
def ngrams(sentence, n=1, lc=True):
    ngram_l = []
    sentence = sentence.lower()
    for i in range(len(sentence) - n + 1):
        ngram = sentence[i:i+n]
        ngram_l.append(ngram)
    return ngram_l

def all_ngrams(sentence, max_ngram=3, lc=True):
    all_ngram_list = []
    for i in range(1, max_ngram + 1):
        all_ngram_list += [ngrams(sentence, n=i, lc=lc)]
    return all_ngram_list

MAX_CHARS = 521
MAX_BIGRAMS = 1031
MAX_TRIGRAMS = 1031
MAXES = [MAX_CHARS, MAX_BIGRAMS, MAX_TRIGRAMS]

def reproducible_hash(string):
    h = hashlib.md5(string.encode("utf-8"), usedforsecurity=False)
    return int.from_bytes(h.digest()[0:8], 'big', signed=True)

def hash_ngrams(ngrams, modulos):
    hash_codes = []
    for ngram_list, modulo in zip(ngrams, modulos):
        codes = [(reproducible_hash(x) % modulo) for x in ngram_list]
        hash_codes.append(codes)
    return hash_codes

def calc_rel_freq(codes):
    cnt = Counter(codes)
    total = sum(cnt.values())
    for k in cnt:
        cnt[k] /= total
    return cnt

MAX_SHIFT = []
for i in range(len(MAXES)):
    MAX_SHIFT += [sum(MAXES[:i])]

def shift_keys(dicts, MAX_SHIFT):
    new_dict = {}
    for i, ngrams_d in enumerate(dicts):
        for k, v in ngrams_d.items():
            new_dict[k + MAX_SHIFT[i]] = v
    return new_dict

def build_freq_dict(sentence, MAXES=MAXES, MAX_SHIFT=MAX_SHIFT):
    hngrams = hash_ngrams(all_ngrams(sentence), MAXES)
    fhcodes = map(calc_rel_freq, hngrams)
    return shift_keys(fhcodes, MAX_SHIFT)

# --- load models ---
clf = joblib.load("Lab4/nld.joblib")
vectorizer = joblib.load("Lab4/nld_vectorizer.joblib")
idx2lang = joblib.load("Lab4/nld_lang_codes.joblib")

input_dim = len(vectorizer.vocabulary_)
nbr_classes = len(idx2lang)

model = nn.Sequential(
    nn.Linear(input_dim, 50),
    nn.ReLU(),
    nn.Linear(50, nbr_classes)
)
model.load_state_dict(torch.load("Lab4/nld.pth", map_location="cpu"))
model.eval()

# --- prediction function ---
def detect_lang(src_sentence):
    src_sentence = [src_sentence]
    X_test = vectorizer.transform(map(build_freq_dict, src_sentence))
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    Y_logits = model(torch.Tensor(X_test))
    pred_languages = torch.argmax(Y_logits, dim=-1).tolist()
    return list(map(idx2lang.get, pred_languages))[0]

# --- Gradio UI ---
with gr.Blocks(title="Tonnys language detector") as demo:
    gr.Markdown("# Tonnys language detector")
    with gr.Row():
        with gr.Column():
            src_sentence = gr.Textbox(
                label="Text", placeholder="Write your text...")
        with gr.Column():
            tgt_sentence = gr.Textbox(
                label="Language", placeholder="Language will show here...")
    btn = gr.Button("Guess the language!")
    btn.click(fn=detect_lang, inputs=[src_sentence], outputs=[tgt_sentence])

demo.launch()