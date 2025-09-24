import os
from zipfile import ZipFile
import requests
import regex as re
import tqdm as tqdm
import math
from collections import Counter
import sentencepiece as spm


# Dataset
# Parameters for Selma dataset, från labben
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

#Train the model, detta är vår BPE bibliotek då som heter Sentence Piece som basically ska funka som våran
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
# Implementerat egen
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
    return set(list(corpus_l))

print(sorted(initial_vocabulary(corpus_l)))

print()

print('New assignment')
# Counting

# Write your code here, min egna
def pair_count(corpus_l):
    pairs = {}

    for i in range(len(corpus_l)-1):
        first = corpus_l[i]
        second = corpus_l[i+1]

        # Kollar om paret är olaglig, annars hoppar vi över iterationen till nästa för att inte ha med de i räkningen
        if first != '\u2581' and second.startswith('\u2581'):
            continue

        # Lägger in de i pair listan och incrementa de
        pair = first, second
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

vocabulary = initial_vocabulary(corpus_l)

print(len(vocabulary))

# Add your most frequent pair to the vocabulary after one iteration as well as the merge operation in the merge_ops list. merge_ops will contain the merge operations in the order you created them.

merge_ops = []

vocabulary.add(pair_tuple)

merge_ops.append(pair_tuple)

print(len(vocabulary))

print(merge_ops)

# Eget
def merge_bigrams(corpus_l, pair):
    first, second = pair
    new_corpus = []
    skip = False

    for i in range(len(corpus_l)):
        if skip == True:  # hoppa över om vi redan har mergat
            skip = False
            continue

        # Om det fortfarande går och loopa, kolla om detta och nästa bildar paret om det matchar vår par som vi vill räkna ihop
        if i < len(corpus_l) - 1 and corpus_l[i] == first and corpus_l[i+1] == second:
            new_corpus.append(first + second)
            skip = True  # hoppa över nästa symbol
        else:
            new_corpus.append(corpus_l[i])

    return new_corpus

corpus_test = ['▁', 'D', 'e', '▁', 's', 'e', 'n', 'a', 's', 't']
test = merge_bigrams(corpus_test, ('e', 'n'))
print(test)

print()

print(merge_bigrams(merge_bigrams(corpus_test, ('e', 'n')), ('s', 'en')))

# Byte Pair Encoding (BPE): Building the Vocabulary

# Algorithm 1 following Bostrom and Durrett, våran BPE modell
# BPE används för att slå ihop vokabulär, hämta de mest vanligaste sammansatta paret och lägga de i både merge_ops och vocabulary
def BPE(corpus_l, k):
    vocabulary = initial_vocabulary(corpus_l)
    merge_ops = []
    
    for _ in range(k): 
        
        pairs = pair_count(corpus_l)
        if not pairs:
            break
        
        # Mest frekventa paret
        most_freq_pair = Counter(pairs).most_common(1)[0][0]

        # Merga två tecken
        corpus_l = merge_bigrams(corpus_l, most_freq_pair)

        # Lägg till nytt subword i vocabulary
        new_symbol = ''.join(most_freq_pair)
        vocabulary.add(new_symbol)

        # Lägg till i merge_ops
        merge_ops.append(most_freq_pair)

    return vocabulary, merge_ops

vocabulary, merge_ops = BPE(corpus_l, 50)
print()
print(merge_ops)
print('text')
merge_ops_text = ''
for pair in merge_ops:
    merge_ops_text += ''.join(pair) + ' '  # konvertera tuple till str och lägg till mellanslag, basically gör om till text
print(merge_ops_text)


print(len(vocabulary), len(merge_ops))

print()

print(vocabulary)

# Tokenisera bpe, och sedan jämför vi med pythons sentencepiece
def tokenize_bpe(corpus, merge_ops):
    tokens = list(corpus)

    for pair in merge_ops:
        tokens = merge_bigrams(tokens, pair)

    return tokens

tokens_bpe = tokenize_bpe(corpus, merge_ops)
print(tokens_bpe)

tokens_sp = sp.encode(corpus_raw, out_type=str)
print(tokens_sp)

#
print("Identical?")
for i, (a, b) in enumerate(zip(tokens_sp, tokens_bpe)):
    if a != b:
        print(i, a, b)
print("Yes")

# Unigram Language Model

# Pythons bibliotek där vi ska jämföra

spm.SentencePieceTrainer.train(
    '--input=Lab3/Selma/herrgard.txt --model_prefix=Lab3/m --vocab_size=116 --user_defined_symbols=0,1,2,3,4,5,6,7,8,9')

sp = spm.SentencePieceProcessor()
print(sp.load('Lab3/m.model'))

print(sp.encode('senare', out_type=str))

print(sp.encode('H', out_type=str))

print(sp.encode('œ', out_type=str))

print([sp.id_to_piece(i) for i in range(30)])

# Unigram Probabilities

# Funkar via att man räknar sannolikheten för alla möjliga subword och tar bort tokens gradvis efteråt via minimeringen av sannolikhetsförlust
def unigram_lm(tokenized_corpus):
   unigram_probs = dict()

   #Returnerar t.ex nåt sånt här {'▁': 3, 'a': 2, 'l': 2, 'S': 1, 'e': 1, 'm': 1, 'L': 1, 'g': 1, 'er': 1, 'ö': 1, 'f': 1}
   count = Counter(tokenized_corpus)
   lengthOfCorpus = len(tokenized_corpus)

   for token, freqOfWord in count.items():
    probability = freqOfWord/lengthOfCorpus
    unigram_probs[token] = math.log(probability)   # naturlig log
   
   return unigram_probs

tokenized_corpus = tokenize_bpe(corpus, merge_ops)
a = tokenized_corpus[:15]

print('ex')
print(a)

#print(corpus)

unigram_logprobs = unigram_lm(tokenized_corpus)
b = sorted(unigram_logprobs.items(), key=lambda x: -x[1])

print(b)
# Har inte tagit bort något än subtoken som det står..

for token in vocabulary:
    if token not in unigram_logprobs:
        print(token)
        unigram_logprobs[token] = -1000

print(len(unigram_logprobs))

# Unigram Tokenization

import functools

# Från labbinstruktionerna
def tokenize_lm(char_seq, unigram_probs):
    # Use one of the two cache functions below to have a faster answer:
    # @functools.lru_cache(maxsize=2**10)
    @functools.cache
    # Available from Python 3.9
    # The arguments of the cached function must be hashable that's why we define an inner cacheable function
    def __tokenize_lm(char_seq):
        if not char_seq:
            return 0.0, []
        splits = [(char_seq[:i], char_seq[i:])
                  for i in range(1, len(char_seq) + 1)]
        candidates = []
        # print(splits)
        for start, rest in splits:
            if start in unigram_probs:  # Else the probability is 0 as well as the product.
                start_prob = unigram_probs[start]
            else:
                start_prob = -1000
            rest_prob, rest_list = __tokenize_lm(rest)
            candidates.append(
                (start_prob + rest_prob, [start] + rest_list))
        # Uncomment the two next lines to see the recursion
        # print('\tCands.', candidates)
        # print('\t-> Max.:', max(candidates))
        return max(candidates)
    return __tokenize_lm(char_seq)


c = tokenize_lm('▁senare', unigram_logprobs)
print(c)

d = tokenize_lm('▁H', unigram_logprobs)
print(d)

# Text Tokenization with Unigrams

re_token = '▁[^▁]+'

e = list(re.finditer(re_token, '▁kdlskdls▁ldösldös▁lödsldö'))
print(e)

# Write your code
def tokenize_text_lm(corpus, unigram_probs):
    tokenized_corpus = []
    corpus_prob = 0.0
    for word in re.finditer(re_token, corpus):
        match = word.group()
        word_prob, token = tokenize_lm(match, unigram_probs)
        '''
        append()
            stoppar in hela listan som ett paket.

            x = [0]
            x.append([1, 2, 3])
            print(x)   # [0, [1, 2, 3]]

        extend()
            tar ut alla saker i listan och lägger in dem en och en.
            Lägger till varje element i den givna listan separat i listan.
            
            x = [0]
            x.extend([1, 2, 3])
            print(x)   # [0, 1, 2, 3]
        '''
        tokenized_corpus.extend(token)
        corpus_prob += word_prob

    return corpus_prob, tokenized_corpus

init_loglikelihood, tokens = tokenize_text_lm(
    corpus, unigram_logprobs)

print(init_loglikelihood, tokens[:10])

# Vocabulary Selection

print(vocabulary)

vocabulary_sorted = sorted(vocabulary, key=lambda w: -unigram_logprobs[w])
print(vocabulary_sorted[:10])

print(corpus_l[0:2])

print(len(merge_ops))


tokenized_corpus = tokenize_bpe(corpus, merge_ops)
unigram_logprobs = unigram_lm(tokenized_corpus)
print(unigram_logprobs)

# Allt detta under är kopierade från instruktionerna

def tokenize_text_lm_2(corpus, unigram_logprobs):
    tokenized_corpus = []
    corpus_prob = 0.0
    for word in re.finditer(re_token, corpus):
        prob, tokens = tokenize_lm(word.group(), unigram_logprobs)
        tokenized_corpus += tokens
        corpus_prob += prob
    return corpus_prob, tokenized_corpus


unigram_logprobs_c = unigram_logprobs.copy()
for i in tqdm.tqdm(range(5)):
    init_loglikelihood, tokens = tokenize_text_lm_2(corpus, unigram_logprobs_c)
    unigram_logprobs_c = unigram_lm(tokens)
    print(init_loglikelihood)

    print(len(unigram_logprobs_c))


print(sorted(unigram_logprobs_c.items(), key=lambda x: -x[1]))

print(tokenize_lm('▁senare', unigram_logprobs_c))


print()

logloss_word = []
for i, word in enumerate(tqdm.tqdm(vocabulary_sorted)):
    if len(word) == 1:
        continue
    # We store the word probability to be able restore it
    prob_word = unigram_logprobs_c.pop(word)
    loglikelihood, tokens = tokenize_text_lm_2(corpus, unigram_logprobs_c)
    log_loss = loglikelihood - init_loglikelihood
    logloss_word += [(log_loss, word)]
    # we restore the probability
    unigram_logprobs_c[word] = prob_word


print(sorted(logloss_word, reverse=True))
# Eget
out_candidate = max(logloss_word, key=lambda x: x[0])[1]  # x[0] jämför logglossen och returnerar ordet [1] då den är lagrad som: t.ex (-9622.446649050573, '▁och')
print()
print("Our candidate",out_candidate)

print("Tokenization without the subwords")
# The tokenization without the subword

print(vocabulary_sorted)

vocabulary_sorted.remove(out_candidate)
# print(vocabulary_sorted) # Efter att elementet/minsta har tagits bort

# Labbens instruktioner
unigram_probs_c = unigram_logprobs.copy()
unigram_probs_c.pop(out_candidate)
for i in range(5):
    new_loglikelihood, tokens = tokenize_text_lm_2(corpus, unigram_probs_c)
    unigram_probs_c = unigram_lm(tokens)
    print(new_loglikelihood)

print()
print(init_loglikelihood) #  -494583.24282243924

print(new_loglikelihood) # 494534.0921189297

print(tokenize_text_lm(corpus, unigram_probs_c))
