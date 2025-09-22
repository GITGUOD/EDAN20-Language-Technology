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

'''
Read these two sections, important for assignment:

Section 3.1 of Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (https://arxiv.org/pdf/1804.10959.pdf) by Kudo (2018).
Section 2, algorithm 1 of Byte Pair Encoding is Suboptimal for Language Model Pretraining (https://aclanthology.org/2020.findings-emnlp.414.pdf) by Bostrom and Durrett (2020).
In your report, in a Method and program structure section, you will introduce your program with a summarization (10 to 15 lines or so) with your own words of the byte-pair encoding (BPE) algorithm as described by Kudo (2018)
and Bostrom and Durrett (2020) (Only BPE and not the unigram language model so far).
'''
print()
print('BPE programming')
print()

# BPE programming

with open(FILE_PATH, encoding='utf8') as f:
    corpus = f.read().strip()

if not corpus.startswith('\u2581'):
    corpus = '\u2581' + corpus

corpus = re.sub(r'\s+', '\u2581', corpus)
print(corpus[:100])

# Initial Vocabulary

#We create a second dictionary to count the subword tokens. At each iteration, the keys will store the subtokens.


corpus_l = list(corpus)

print(corpus_l)

print(corpus_l[:15])

print()
#Extract the set of characters that will serve as initial subword tokens: Write a statement to extract the set of all the characters from corpus_l

char_set = set(corpus_l)
print(char_set)

sortedSet = sorted(char_set)

print()

print(sortedSet)

print(len(char_set))

# Write your code here
def initial_vocabulary(corpus_l):
    return sorted(set(list(corpus_l)))

print(initial_vocabulary(corpus_l))

print()

print('New assignment')
# Counting

# Write your code here
def pair_count(corpus_l):
    pairs = {}

    for i in range(len(corpus_l)-1):
        #hoppar över _
        if corpus_l[i] != '\u2581' and corpus_l[i+1] == '\u2581':
            continue

        pair = corpus_l[i], corpus_l[i+1]
        if pair in pairs:
            pairs[pair] += 1
        else:
            pairs[pair] = 1
 
    return pairs

pairs = pair_count(corpus_l)

print(Counter(pairs))

most_freq_pair = Counter(pairs).most_common(1)[0] 
pair_tuple, count = most_freq_pair

print(pair_tuple)

pair = ''.join(pair_tuple)
print(pair)

# The First Iteration

