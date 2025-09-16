import bz2
import math
import os
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


# Non-letter nor the following punctations
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

text = '<s>' + sentence_markup + '</s>' # We insert the s or s:es due to the substitution only works between sentences, not at the start nor end
#print(text)

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

    sentence_boundaries = r"([.!?])(\s+)(\p{L})"

    sentence_markup = r"\1 </s>\n<s> \3"

    text = re.sub(sentence_boundaries, sentence_markup, text)

    p_removal = r"[.;:?!]"

    text = re.sub(p_removal, '', text)

    Sduplicates = r'\s+'

    text = re.sub(Sduplicates, ' ', text)


    text = " <s> " + text + " </s> "

    return text

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
print(list(frequency_bigrams.items())[:20])

# In the report, tell what is the possible number of bigrams and their real number?
# Explain why such a difference. What would be the possible number of 4-grams.