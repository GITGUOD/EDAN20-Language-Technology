import os
from zipfile import ZipFile
import requests
import regex as re
import tqdm as tqdm
import math
from collections import Counter
import sentencepiece as spm


# Dataset
# Parameters for Selma dataset
SELMA_URL = "https://github.com/pnugues/ilppp/raw/master/programs/corpus/Selma.zip"

SELMA_FILES = [
    os.path.join("Lab3", "Selma", fname)
    for fname in
    [
        "bannlyst.txt",
        "gosta.txt",
        "herrgard.txt",
        "jerusalem.txt",
        "kejsaren.txt",
        "marbacka.txt",
        "nils.txt",
        "osynliga.txt",
        "troll.txt"
    ]
]


def download_and_extract_selma():
    """Downloads and unpacks Selma.zip"""

    # Download if not all files exist
    req = requests.get(SELMA_URL, stream=True)
    if req.status_code != 200:
        print("Failed to download file, got status: " + req.status_code)
        req.close()
    else:
        with open("Selma.zip", "wb") as fd:
            written = 0
            for chunk in req.iter_content(chunk_size=65536):
                fd.write(chunk)
                written += len(chunk)
                print("Downloading: %d bytes written to Selma.zip" % written)

        print("Selma.zip donwnloaded.")
        req.close()

        selma_zipfile = ZipFile("Selma.zip")
        selma_files_to_extract = [zi for zi in selma_zipfile.filelist if not zi.filename.startswith(
            "__") and zi.filename.endswith(".txt")]
        for zi in selma_files_to_extract:
            selma_zipfile.extract(zi)
            print("Extracted: " + zi.filename)

        print("Done!")


# If not all path exists (all are true), then download
if not all([os.path.exists(fname) for fname in SELMA_FILES]):
    download_and_extract_selma()
else:
    print("Selma has been downloaded.")

SELMA_FILES

# FILE_PATH = '../../corpus/Selma.txt'
FILE_PATH = 'Lab3/Selma/herrgard.txt'

with open(FILE_PATH, encoding='utf8') as f:
    corpus_raw = f.read().strip()

print(corpus_raw[:100])

#pip install sentencepiece

#Train the model
spm.SentencePieceTrainer.train('--input=Lab3/Selma/herrgard.txt --model_prefix=Lab3/m --vocab_size=116 --model_type=BPE --user_defined_symbols=0,1,2,3,4,5,6,7,8,9 ')

sp = spm.SentencePieceProcessor()
sp.load('Lab3/m.model')

#Tokenize, consist of subword ids
print(sp.encode('Selma Lagerlöf'))

print(sp.id_to_piece(63), sp.id_to_piece(96))

print([sp.id_to_piece(i) for i in range(30)])

print(sp.encode('Selma Lagerlöf', out_type=str))

print(sp.encode('123', out_type=str))

print(sp.encode(corpus_raw, out_type=str))