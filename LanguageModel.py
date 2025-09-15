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

def tokenize(text):
    words = re.findall(r'\p{L}+', text)
    return words

words = tokenize(corpus)
print(words[:10])

# Count the number of unique words in the original corpus and when setting all the words in lowercase

def countUniqueWordsOfText(text):
    dict = dict()
    for word in text:
        dict.append(word.lower())
    return dict.len()
print(countUniqueWordsOfText(corpus))





