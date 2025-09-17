import bz2
import math
import os
from collections import Counter
import regex as re
import requests
import sys
from zipfile import ZipFile
import Concord as con
import sys, importlib

corpus = open('Lab 2/Selma.txt', encoding='utf8').read()

pattern = 'Nils Holgersson'
width = 25

# con.concordance('Lab 2/Selma.txt', pattern, width)

def tokenize(regex, text):
    words = re.findall(regex, text)
    return words

words = tokenize(r'\p{L}+', corpus)
# print(words[:10])

# Count the number of unique words in the original corpus and when setting all the words in lowercase

def countUniqueWordsOfText(text):
    #tokenize words first as text is just a string:
    words = tokenize(r'\p{L}+', text)
    unique_words_L = set()
    unique_words_H = set()

    for word in words:
        unique_words_L.add(word.lower())
    for word in words:
        unique_words_H.add(word)
    return len(unique_words_L), len(unique_words_H)
# print(countUniqueWordsOfText(corpus))

# Segmenting a corpus
# You will write a program to tokenize your text, insert <s> and </s> tags to delimit sentences, and set all the words in lowercase letters.
# In the end, you will only keep the words.


# Non-letter, alltså att vi inte tar någon av dom under
nonletter = r'[^\p{L}+\.;:?!]'

def clean(text):
    cleanedText = re.sub(nonletter, ' ', text)
    return cleanedText

test_para = 'En gång hade de på Mårbacka en barnpiga, som hette Back-Kajsa. \
Hon var nog sina tre alnar lång, hon hade ett stort, grovt ansikte med stränga, mörka drag, \
hennes händer voro hårda och fulla av sprickor, som barnens hår fastnade i, \
när hon kammade dem, och till humöret var hon dyster och sorgbunden.'
test_par = clean(test_para)
#print(test_par)

# print(corpus[:200])

# print(clean(corpus[:200]))

# Segmenter
# Detecting sentence boundaries
#\s+ stands for space and + stands for more if there are more than one space
sentence_boundaries = r"([.!?])(\s+)(\p{Lu})"

sentence_markup = r"\1 </s>\n<s> " # The numbers matches the above sentence boundaries. </s> comes after .!? etc due to it being first


sentence_markup = re.sub(sentence_boundaries, sentence_markup, test_para)

# Applying substitution:
text = sentence_markup
#print(text)

text = '<s>' + sentence_markup + '</s>'

# Replace the space duplicates with one space and remove the punctuation signs. For the spaces, use the Unicode regex. You may use \s though.

#Removing space duplicates:
Sduplicates = r'\s+'

textWithoutDuplicateSpaces = re.sub(Sduplicates, ' ', text)

#print(textWithoutDuplicateSpaces)

# Removing punctuation signs

# Match punctuation characters: . ; : ? !
punctuationRemoval = r"[.;:?!]"

textWithoutPunctuation = re.sub(punctuationRemoval, '', textWithoutDuplicateSpaces)
# print(textWithoutPunctuation)


def segment_sentences(text):
    text = text.lower()

    sentence_boundaries = r"([.!?])(\s+)(\p{L})" # Om ordet slutar med .!?, har mellan slag efter och har ett stort bokstav efter så slutar meningen

    sentence_markup = r"\1 </s>\n<s> \3"

    text = re.sub(sentence_boundaries, sentence_markup, text) # Substituerar sentence_boundaries med <s> och <\s> på slutet

    p_removal = r"[.;:?!]" # Byter ut alla dessa tecken med '' under, vi tar alltså bort de

    text = re.sub(p_removal, '', text)

    Sduplicates = r'\s+' #Tar bort extra mellanslag

    text = re.sub(Sduplicates, ' ', text)


    text = " <s> " + text + " </s> " # Sätter s i början och slutet av texten eftersom våran kod inte tar hänsyn i början och på slutet

    return text

# Detta funkar dock inte om vi har undantag så som Namn som börjar med storbokstav eller förkortningar

#print(segment_sentences(test_para))


# Accuracy? Hm, jag tror den har ganska bra accuracy men tar inte hänsyn till t.ex citat eller liknande så som förkortningar etc.

def segmenting_corpus(text):
    text = clean(text)
    text = segment_sentences(text)
    return text
#print("new line")
s = segmenting_corpus(corpus[-557:])

#print(s)

# Fel i uppgiften? Vi får en extra mening i början.

# List of Words
def createAListOfWords(text):
    '''
    Split metoden gör om strängen till en lista
    '''
    return text.split()

newWords = createAListOfWords(segmenting_corpus(corpus))
# print(newWords)


#print(segmenting_corpus(corpus[-557:]))

# Counting unigrams and bigrams

# Unigrams
def unigrams(words):
    frequency = {}
    for i in range(len(words)):
        if words[i] in frequency:
            frequency[words[i]] += 1
        else:
            frequency[words[i]] = 1
    return frequency

frequency = unigrams(newWords)
list(frequency.items())[:20]
print("unigrams")

print(list(frequency.items())[:20])

print()
print("bigrams")


# Bigrams
def bigrams(words):
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i + 1]))
    frequency_bigrams = {}
    for i in range(len(words) - 1):
        if bigrams[i] in frequency_bigrams:
            frequency_bigrams[bigrams[i]] += 1
        else:
            frequency_bigrams[bigrams[i]] = 1
    return frequency_bigrams
frequency_bigrams = bigrams(newWords)
#print(list(frequency_bigrams.items())[:20])

# In the report, tell what is the possible number of bigrams and their real number?
# Explain why such a difference. What would be the possible number of 4-grams.

# Write your code
def unigram_lm(frequency, sent_words):
    #Frequency räknar antalet gånger ordet/unigrammet kommer fram i texten corpus

    frequency_of_words = 0
    for count in frequency.values():
        frequency_of_words += count

    """
    Computing the sentence probs with a unigram model.
    """

    print('=====================================================')
    print("wi \t C(wi) \t #words \t P(wi)")
    print('=====================================================')

    entropy = 0
    sentenceP = 1.0
    for wi in sent_words:
        countingWord = frequency.get(wi, 0)
        probabilityOfWord = countingWord/frequency_of_words # räknar sannolikheten av ett ord genom att ta det ordet och dela på antalet ord
        sentenceP = sentenceP*probabilityOfWord
        print(f"{wi}\t{countingWord}\t{frequency_of_words}\t{probabilityOfWord}")

    #According to the chapter 10 lesson
    # entroy = H(L) = 1/(-n) * log2(P(w1,...,Wn)) A measure of uncertainty or unpredictability in your language model. Räknar alltså osäkerheten
    # Perplexity är 2^H(L), Exponential of the entropy. A standard way to measure how well your model predicts a sentence, större sannolikhet om perplexity är hög etc. Hur väl den gissar en mening
    # geo_mean_prob = probabilities of all words in the sentence

    n = len(sent_words)
    entropy = - (1 / n) * math.log2(sentenceP)

    perplexity = 2**entropy
    geo_mean_prob = sentenceP**(1/len(sent_words))

    print('=====================================================')
    print("Prob. unigrams:\t", sentenceP)
    print("Geometric mean prob.:", geo_mean_prob)
    print("Entropy rate:\t", entropy)
    print("Perplexity:\t", perplexity)

    return perplexity


sentence = 'det var en gång en katt som hette nils </s>'
sent_words = sentence.split()
print(sent_words)

perplexity_unigrams = unigram_lm(frequency, sent_words)

perplexity_unigrams = int(perplexity_unigrams)
print(perplexity_unigrams)


print()
# Bigrams
print('bigram')
def bigram_lm(frequency, frequency_bigrams, sent_words):

    frequency_of_words = 0 #Number of words
    for count in frequency.values():
        frequency_of_words += count

    """
    Computing the sentence probs with a bigram model.
    """

    print('=====================================================')
    print("wi \t wi+1 \t Ci,i+1 \t C(i) \t P(wi+1|wi)")
    print('=====================================================')

    entropy = 0
    sentenceP = 1.0
    n = len(sent_words)

    for i in range(1, n):
        wi_1, wi = sent_words[i-1], sent_words[i]
        bigram = (wi_1, wi)
        count_bigram = frequency_bigrams.get(bigram, 0)
        count_prev = frequency.get(sent_words[i-1], 0)

        backoff = ""

        if count_bigram > 0:
            prob = count_bigram / count_prev
        else:

            prob = frequency.get(wi, 0) / sum(frequency.values())
            backoff = f"*backoff: {prob}" # Ifall probability blir 0, och om vi har sett alla kombinationer

        sentenceP = sentenceP*prob

        print(f"{bigram[0]}\t{bigram[1]}\t{count_bigram}\t{count_prev}\t{prob}\t{backoff}")

    # Samma som unigram men med bigram

    entropy = - (1 / (n-1)) * math.log2(sentenceP)
    perplexity = 2 ** entropy
    geo_mean_prob = sentenceP ** (1/(n-1)) #Formeln ser annorlunda ut för att vi börja räkna bigrams efter det andra ordet

    print('=====================================================')
    print("Prob. unigrams:\t", sentenceP)
    print("Geometric mean prob.:", geo_mean_prob)
    print("Entropy rate:\t", entropy)
    print("Perplexity:\t", perplexity)

    return perplexity

sentenceBi = '<s> det var en gång en katt som hette nils </s>'
#sentenceBi2 = '<s> LU är en av Sveriges äldsta universitet </s>'
#sentenceBi3 = '<s> Det är svårt och bestämma sig vilken maträtt man ska äta varje dag </s>'

sent_wordsBi = sentenceBi.split() # Det är för att bigrams behöver par, annars står ett ord utanför

perplexity_bigrams = bigram_lm(frequency, frequency_bigrams, sent_wordsBi)

perplexity_bigrams = int(perplexity_bigrams)
print(perplexity_bigrams)

# Trigrams
print('Trigrams')

# Stulit från notebooks
def trigrams(words):
    # Create trigrams as tuples
    trigrams = [tuple(words[idx:idx + 3]) for idx in range(len(words) - 2)]
    # Count frequencies
    trigram_freqs = Counter(trigrams)
    return trigram_freqs


frequency_trigrams = trigrams(newWords)
print(frequency_trigrams[('det', 'var', 'en')])  # 330


# Predictions

cand_nbr = 5
starting_text = 'De'.lower()

candidates = []
for (w1, w2), count in frequency_bigrams.items():
    if w1 == '<s>' and w2.startswith(starting_text):
        candidates.append((w2, count))

NewList = []
for i in range(5):
    NewList.append(candidates[i])
NewList.sort(key=lambda x: (-x[1], x[0]))

print(NewList)

current_text = "Det var en ".lower()
t = tokenize(r'\p{L}+', current_text)
print(t)

w1, w2 = t[-2], t[-1]

candidates = []
for (a, b, c), count in frequency_trigrams.items():
    if a == w1 and b == w2:
        candidates.append((c, count))

candidates.sort(key=lambda x: (-x[1], x[0]))

# Tar top 5
next_word_predictions = [c[0] for c in candidates[:5]]

print(next_word_predictions)



current_text = "Det var en g".lower()
tokens = current_text.split()
w1, w2 = tokens[-3], tokens[-2]  # second-to-last and last full words
partial = tokens[-1]                      # g

candidates = []
# Kollar alla tre ord, 
for (a, b, c), count in frequency_trigrams.items():
    if a == w1 and b == w2 and c.startswith(partial):
        candidates.append((c, count))


candidates.sort(key=lambda x: (-x[1], x[0])) #sorterar störst först baserat på värdet
current_word_predictions_2 = [c[0] for c in candidates[:5]]
print(current_word_predictions_2)
