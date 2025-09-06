# Lab 1, we will create an indexer, index consists of rows with one word per row and
# the list of files and positions where this words occur. Such a row is called a posting list.

import bz2
import math
import os
import pickle
import regex as re
import requests
import sys
import time
from zipfile import ZipFile

# We have downloaded Selmas file

'''
Write a program that reads one document file_name.txt and outputs an index file: file_name.idx:

The index file will contain all the unique words in the document, where each word is associated with the list of its positions in the document.
You will represent this index as a dictionary, where the keys will be the words, and the values, the lists of positions
As words, you will consider all the strings of letters that you will set in lower case. You will not index the rest (i.e. numbers, punctuations, or symbols).
To extract the words, use Unicode regular expressions. Do not use \w+, for instance, but the Unicode equivalent.
The word positions will correspond to the number of characters from the beginning of the file. (The word offset from the beginning)
You will use the finditer() method to find the positions of the words. This will return you match objects, where you will get the matches and the positions with the group() and start() methods.
You will use the pickle package to write your dictionary in an file, see https://wiki.python.org/moin/UsingPickle.
'''

#ToDo:

# Make the program so that it can open and read one document file: Finished

text = open('Selma/bannlyst.txt').read()

# 1. Index file will contain unique words. Varje ord kommer ha en lista med positioner i dokumentet så som föreläsningen med index antar jag typ (11, 34).
# Dictionary ska användas där nycklar kommer att vara ord och värdena ska vara positioner vilket gör att orden blir unika.
# ord kommer vara i lowercase. Endast ord i listan och inte tecken, symboler eller punkter etc
# När vi extractar orden, använd unicode regular expressions (regex) ingen \w+.
# The word positions will correspond to the number of characters from the beginning of the file. (The word offset from the beginning) 
# finditer() kommer att användas för att hitta positioner, matcha objekt, where you will get the matches and the positions with the group() and start() methods.

# ALLT ovan är fixat


index = dict() # Skapa en dictionary

for match in re.finditer(r'\p{L}+', text): # För varje match i texten, L för letters då vi inte vill ha andra symboler etc. Lägga till + för att få med hela ordet
    word = match.group().lower() # Varje match kör vi till lower case
    position = match.start() # hämtar positionen av ordet
    if word not in index: # Lägga till ordet i vår dictionary för första gången
        index[word] = []
    index[word].append(position) # Appenda positionen i listan med vår nyckel

pickle.dump(index, open("index.p", "wb"))

#Output index file via pickle
# Klart!


# Öppna filen igen:

text = pickle.load(open("index.p", "rb"))

line = "framkomma att älska förett liv"

regex = re.findall(r"f.a", line) #Försöker ta fram ordet 'framkomma'


print(index)
print(regex)
