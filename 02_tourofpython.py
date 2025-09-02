# A Tour Of Python

# Creating Variables
'''
    a = 1
    b = 2
    c = a / (b+1)
    text = 'Result:'
    print(text, c)
'''


'''
    # Elementatry flow control
    # The for loop
    for i in [1, 2, 3, 4, 5, 6]:
        print(i)
    print('Done')
'''

# Conditionals 

'''
    for i in [1, 2, 3, 4, 5, 6]:
        if i % 2 == 0:
            print('Even:', i)
        else:
            print('Odd:', i)
    print('Done')
'''

# Strings
'''
    iliad_opening = "Sing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans."
    print(iliad_opening)
'''
iliad_opening = "Sing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans."
'''
alphabet = 'abcdefghijklmnopqrstuvwxyz'
var1 = alphabet[0]
var2 = alphabet[1]
var3 = alphabet[25]
print(var1, var2, var3)

alphabet[-1]
print(alphabet[-26])
#print(alphabet[27])
print(len(alphabet))

alphabet[0] = 'b'  # throws an error, as strings are immutable

'''


'''
# String Operations and Functions
extendedWord = 'abc' + 'def'
print(extendedWord)

multiplyingWord = 'abc' * 3
print(multiplyingWord)

# String functions
newWord = extendedWord.join(['abc', 'def', 'ghi'])
print(newWord)
print(' '.join(['abc', 'def', 'ghi']))
print(''.join(['abc', 'def', 'ghi']))
print(', '.join(['abc', 'def', 'ghi']))

'''

'''
# upper() and lower()
accented_e = 'eéèêë'
accented_e.upper()  # 'EÉÈÊË'
print(accented_e)

alphabet = 'abcdefghijklmnopqrstuvwxyz'
findingWord = alphabet.find('xyz')
print(findingWord)
print(alphabet.find('é'))  # -1, not found

alphabet.replace('abc', 'aBC')
'''

# Continuation: Program to extract vowels
'''
# Extracting words = ''
iliad_opening = "Sing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans."

initial = ''
for word in iliad_opening:
    if( word in 'aeiou'):
        initial = initial + word
print(initial)
'''

'''
# Slices
alphabet = 'abcdefghijklmnopqrstuvwxyz'
first = alphabet[0:3]
second = alphabet[:3] #Samma som first
third = alphabet[3:6] # index 3 till 5 så: 'def'
fourth = alphabet[-3:] # 'xyz'
fifth = alphabet[10:-10]  # 'klmnop'


print(first)  # 'abc'
print(second) # 'abc'
print(third)  # 'def'
print(fourth) # 'xyz'
print(fifth)  # 'klmnop'
print(alphabet[:]) # Hela alfabetet

'''

# The whole string
'''
alphabet = 'abcdefghijklmnopqrstuvwxyz'
i = 10
alphabet[:i] + alphabet[i:]  # 'abcdefghijklmnopqrstuvwxyz'
print(alphabet[:i] + alphabet[i:]) # 'abcdefghijklmnopqrstuvwxyz'

print(alphabet[0::2]) # 'acegikmopqsuwy'
'''

# Special Characters

'''
print('Python\'s strings') #Samma som under
print("Python's strings")
print('\N{Commercial AT}') # @
print('\x40') # @, Använder UTF-8 standard
print('\u0152') # 'Œ'

# r som prefix för att behandla backslashes som vanliga tecken/karaktärer
print(r'\N{COMMERCIAL AT}')  # '\\N{COMMERCIAL AT}'
print(r'\x40') # @, Använder UTF-8 standard
print(r'\u0152') # 'Œ'

'''
# Formatting strings
'''
begin = 'my'
print('{} string {}'.format(begin, 'is empty')) # 'my string is empty'
'''

# Data identifies and types
alphabet = 'abcdefghijklmnopqrstuvwxyz'

'''
print(12)
print(type(12)) # <class 'int'>
print(id(12)) # Jag fick inte outputen 4305677768

a = 12
print(id(a))
print(type(a))
print(type(12.0))  # <class 'float'>
print(type(True))  # <class 'bool'>
print(type(1 < 2))  # <class 'bool'>
print(type(None))  # <class 'NoneType'>
b = '12'
print(id(b))
print(type('12'))
print(id(alphabet))
print(type(alphabet))  # <class 'str'>
'''

# Type conversions
'''
a = int('12')  # 12
b = str(12)  # '12'
c = int('12.0') # Ger oss ERROR
print(c)  # 12
int(alphabet)  # Ger ValueError

'''

'''
print(int(True))  # 1

int(False)  # 0

bool(7)  # True

bool(0)  # False

bool(None)  # False

'''

#Data structures

'''
list1 = [] #Empty list
list1 = list() #Another way to create an empty list
list2 = [1,2,3] # List which contains three integers
print(type(list2))

print(list2[1])

list2[1] = 8 # Nu innehåller listan [1, 8, 3]

#list2[4] #Index error

var1 = 3.14
var2 = 'my string'

list3 = [1, 3.14, 'Prolog', 'myString']
print(list3)
print(type(list3))

#Slices

list3[1:3]  # [3.14, 'Prolog']
list3[1:3] = [2.72, 'Perl', 'Python'] # Replaces two elements with three new ones and list extends
print(list3)  # [1, 2.72, 'Perl', 'Python', 'my string']

#List of lists

list4 = [list2, list3]
# [[1, 8, 3], [1, 2.72, 'Perl', 'Python', 'my string']]
print(list4)

list4[0][1]  # 8

list4[1][3]  # 'Python'
print(list4[1][3])  # 'Python', Så det funkar via att man tar första andra elemtet i vektorn vilket är lista 2 och tar det fjärde elementet i lista2


list5 = list2
[v1, v2, v3] = list5 # Unpacking variables and their values
print([v1, v2, v3])



# List Copy
print(id(list2))
print(list2)
print(list5)
list6 = list2.copy()
print(id(list6))

#Identity and equality
print(list2 == list5)
print(list2 == list6)
print(list2 is list5)
print(list2 is list6)

list2[1] = 2
print(list2)
print(list5)
print(list6)
print(id(list2))

#Deep Copy
print(id(list4.copy()[0]))

import copy
print(id(copy.deepcopy(list4)[0]))

#List operations and functions
print(list3[:])
print(list3[:-1])
print([1, 2, 3] + ['a', 'b'])  # [1, 2, 3, 'a', 'b']
print(list2[:2] + list3[2:-1])
print(list2*2)
print([0.0]*4)

print(len(list2))  # 3

list2.extend([4, 5])  # [1, 2, 3, 4, 5]
print(list2)

list2.append(6)  # [1, 2, 3, 4, 5, 6]
print(list2)

list2.append([7, 8])  # [1, 2, 3, 4, 5, 6, [7, 8]]
print(list2)

list2.pop(-1)  # [1, 2, 3, 4, 5, 6]
print(list2)


list2.remove(1)  # [2, 3, 4, 5, 6]
print(list2)

list2.insert(0, 'a')  # ['a', 2, 3, 4, 5, 6]
print(list2)

print(list5)
print(list6)

'''

#Tuples

'''
Tuple1 = () # Empty tuple
tuple1 = tuple()  # Another way to create an empty tuple
tuple2 = (1, 2, 3, 4)
print(tuple2)
print(tuple2[3])

tuple2[1:4]  # (2, 3, 4)

#tuple2[3] = 8  # Type error: Tuples are immutable
list7 = ['a', 'b', 'c']
tuple3 = tuple(list7)  # conversion to a tuple: ('a', 'b', 'c')
print(tuple3)

print(type(tuple3))  # <class 'tuple'>

list8 = list(tuple2)  # [1, 2, 3, 4]
print(tuple([1]))

print(list((1,)))

tuple4 = (tuple2, list7)  # ((1, 2, 3, 4), ['a', 'b', 'c'])
print(tuple4[0])  # (1, 2, 3, 4),

tuple4[1]  # ['a', 'b', 'c']

print(tuple4[0][2])  # 3

print(tuple4[1][1])  # 'b'

tuple4[1][1] = 'β'  # ((1, 2, 3, 4), ['a', 'β', 'c'])
print(tuple4)
'''

# Sets

'''
set1 = set()  # An empty set
set2 = {'a', 'b', 'c', 'c', 'b'}  # {'a', 'b', 'c'}
print(set2) # A set has no duplicates, simply the unique ones

print(type(set2))

set2.add('d')
print(set2)

set2.remove('a')
print(set2)

list9 = ['a', 'b', 'c', 'c', 'b']
set3 = set(list9)  # {'a', 'b', 'c'}
print(set3)  # A set has no duplicates, simply the unique ones

iliad_chars = set(iliad_opening.lower()) # Set takes the string and divided into characters which then catagorizes unique chars
# The set of unique characters of the iliad_opening string
print(iliad_chars)

print(sorted(iliad_chars))

# Operations

newSet = set2.intersection(set3)  # {'c', 'b'} It takes unique character which both sets contains
# Another way to write it is set2 & set3
print(newSet)

print(set2.union(set3))

set2 | set3 # OR, ta med båda eller alla

set2.symmetric_difference(set3)  # {'a', 'd'}
# Det tar unika tecken som finns i antingen set2 eller set3 men inte i båda


set2.issubset(set3)  # False
# Det betyder att set2 inte är en delmängd av set3
alphabet = 'abcdefghijklmnopqrstuvwxyz'
print(iliad_chars.intersection(set(alphabet)))
# characters of the iliad string that are letters:
# {'a', 's', 'g', 'p', 'u', 'h', 'c', 'l', 'i',
#  'd', 'o', 'e', 'b', 't', 'f', 'r', 'n'}

#Tar med alla unika characters som både finns i iliad och alfabetet


'''

# Dictionaries

'''
wordcount = {}  # We create an empty dictionary
wordcount = dict()  # Another way to create a dictionary
wordcount['a'] = 21  # The key 'a' has value 21
wordcount['And'] = 10  # 'And' has value 10
wordcount['the'] = 18

print(wordcount)

print(type(wordcount))

print(wordcount['a'])  # 21

print(wordcount['And'])  # 10

print('And' in wordcount)  # True

# print(wordcount['is'])  # Key error eftersom den inte finns

# Dictionary functions

wordcount.get('And') # 10

wordcount.get('is', 0)  # 0, returns the value zero as it would otherwise return none as is doesnt exist in the directionary. 

from collections import defaultdict

missing_proof = defaultdict(int)
missing_proof['the']

print(missing_proof['the']) # Here, int is used as the “default factory”. int() with no arguments → 0. So, any missing key will automatically get the value 0.
# so defaultdict is jjust a dictionary but more freedom to choose defaultvalue?

#Dictionary Functions
wordcount.keys()

print(wordcount.keys())  # dict_keys(['a', 'And', 'the'])

print(wordcount.values())  # dict_values([21, 10, 18])
print(wordcount.items())  # dict_items([('a', 21), ('And', 10), ('the', 18)])

# Keys must be immutable
my_dict = {}
my_dict[('And', 'the')] = 3
# my_dict[[1, 2]] = 4  # Type error: list is not hashable

# Counting Letters with a dictionary

letter_count = {}
for letter in iliad_opening.lower():
    if letter in alphabet:
        if letter in letter_count:
            letter_count[letter] += 1
        else:
            letter_count[letter] = 1

print('Iliad')
print(letter_count)

print("newLine")
# Sorting the letters by frequency
for letter in sorted(letter_count.keys()):
    print(letter, letter_count[letter]) 
'''

# Control Structures

# Conditionals

'''

digits = '0123456789'
punctuation = '.,;:?!'
char = '.'

if char in alphabet:
    print('Letter')
elif char in digits:
    print('Number')
elif char in punctuation:
    print('Punctuation')
else:
    print('Other')


# The for... in loop

sum = 0
for i in range(100):
    sum += i
print(sum)  # Sum of integers from 0 to 99: 4950
# Using the built-in sum() function,
# sum(range(100)) would produce the same result.

list10 = list(range(5))  # [0, 1, 2, 3, 4]
print(list10)

for inx, letter in enumerate(alphabet):
    print(inx, letter)

# inx = index, letter = element, and enumerate is just a neat shortcut to keep track of both in one loop.
# enumerate(alphabet) produces pairs: (0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')


# We cannot change an iteration variable in Python
for i in list10:
    if i == 0:
        i = 10
print(list10)    # [0, 1, 2, 3, 4]

#The variable i you use in a for loop is just a copy of the element from the list (or whatever iterable you loop over).
#Changing i won’t change the list itself.

# While loop
sum, i = 0, 0
while i < 100:
    sum += i
    i += 1
print(sum)

sum, i = 0, 0
while True:
    sum += i
    i += 1
    if i >= 100:
        break
print(sum)

# Exceptions

try:
    int(alphabet)
    int('12.0')
except:
    pass
print('Cleared the exception!')

try:
    int(alphabet)
    int('12.0')
except ValueError:
    print('Caught a value error!')
except TypeError:
    print('Caught a type error!')


# lc is for lowercase. It is to set the characters in lowercase
# We define a function with the def keyword:
def count_letters(text, lc=True):
    letter_count = {}
    if lc:
        text = text.lower()
    for letter in text:
        if letter.lower() in alphabet:
            if letter in letter_count:
                letter_count[letter] += 1
            else:
                letter_count[letter] = 1
    return letter_count


# We call the function with it default arguments
od = count_letters(iliad_opening)
for letter in sorted(od.keys()):
    print(letter, od[letter])

print(type(count_letters))


# Documenting Functions

def count_letters(text, lc=True):
    """
    Count the letters in a text
    Arguments:
        text: input text
        lc: lowercase. If true, sets the characters 
        in lowercase
    Returns: The letter counts
    """
    letter_count = {}
    if lc:
        text = text.lower()
    for letter in text:
        if letter.lower() in alphabet:
            if letter in letter_count:
                letter_count[letter] += 1
            else:
                letter_count[letter] = 1
    return letter_count

help(count_letters)

# Pilen är bara retur typ
def count_letters(text: str, lc: bool = True) -> dict[str, int]:
    """
    Count the letters in a text
    Arguments:
        text: input text
        lc: lowercase. If true, sets the characters in lowercase
    Returns: The letter counts
    """
    letter_count = {}
    if lc:
        text = text.lower()
    for letter in text:
        if letter.lower() in alphabet:
            if letter in letter_count:
                letter_count[letter] += 1
            else:
                letter_count[letter] = 1
    return letter_count

'''
# Comprehension and Generators

'''
word = 'acress'
splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
print(splits)

deletes = [a + b[1:] for a, b in splits if b]
print(deletes)

splits = []
for i in range(len(word) + 1):
    splits.append((word[:i], word[i:]))
print(splits)

deletes = []
for a, b in splits:
    if b:
        deletes.append(a + b[1:])
print(deletes)

print("new line")
# Generators

    # Generators are similar to comprehensions, but they create the elements on demand

splits_generator = ((word[:i], word[i:])
                    for i in range(len(word) + 1))

print(splits_generator)

for i in splits_generator:
    print(i)

# We can traverse a generator only once
for i in splits_generator:
    print(i)  # Nothing
print(type(splits_generator))  # <class 'generator'>

# Iterators:

my_iterator = iter('abc')

# We access the next item of iterators with the a next() function:
print(next(my_iterator))
print(next(my_iterator))

print(next(my_iterator))

#print(next(my_iterator)) Error as StopIteration

# zip()

latin_alphabet = 'abcdefghijklmnopqrstuvwxyz'
len(latin_alphabet)  # 26

greek_alphabet = 'αβγδεζηθικλμνξοπρστυφχψω'
len(greek_alphabet)  # 24

cyrillic_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
len(cyrillic_alphabet)  # 33

la_gr = zip(latin_alphabet[:3], greek_alphabet[:3])
print(la_gr)
print(list(la_gr))
print(list(la_gr))  # You can traverse it only once

la_gr_cy = zip(latin_alphabet[:3], greek_alphabet[:3],
               cyrillic_alphabet[:3])
print(la_gr_cy)

la_gr = zip(latin_alphabet[:3], greek_alphabet[:3])  # We recreate the iterator
print(next(la_gr))  # ('a', 'α')
print(next(la_gr))  # ('a', 'α')
print(next(la_gr))  # ('a', 'α')
# print(next(la_gr))  # ('a', 'α') Inget mer och iterera

# We can traverse an iterator only once. To traverse it two or more times, we convert it to a list

la_gr_cy_list = list(la_gr_cy)
# First time

la_gr_cy_list  # [('a', 'α', 'а'), ('b', 'β', 'б'), ('c', 'γ', 'в')]
# Second time, etc.

la_gr_cy_list  # [('a', 'α', 'а'), ('b', 'β', 'б'), ('c', 'γ', 'в')]
list(la_gr_cy)  # []

#Zipping

la_gr_cy = list(zip(latin_alphabet[:3], greek_alphabet[:3],
                    cyrillic_alphabet[:3]))
print(la_gr_cy)

te = list(zip(*la_gr_cy))  # [('a', 'b', 'c'), ('α', 'β', 'γ'), ('а', 'б', 'в')]
print(te)

print(la_gr_cy_list)
print(list(zip(*la_gr_cy_list)))

# zipping is like building a matrix row by row, and unzipping swaps the row with the column?
'''

# Modules

#The math module

'''
import math

math.sqrt(2)  # 1.4142135623730951

math.sin(math.pi / 2)  # 1.0

math.log(8, 2)  # 3.0

print(type(math))

# The statistics module

import statistics as stats

stats.mean([1, 2, 3, 4, 5])  # 3.0 # Average value

stats.stdev([1, 2, 3, 4, 5])  # 1.5811388300841898

#Running the program or importing it

if __name__ == '__main__':
    print("Running the program")
    # Other statements
else:
    print("Importing the program")
'''

    # Other statements

#python namnetPåFilen.py
#så körs filen direkt, och i den filen är:

#python
#Kopiera kod
#if __name__ == '__main__':
#sant (__name__ == "__main__"), så allt inuti den blocken kommer att köras.

#Om du istället importerar filen i en annan fil, t.ex.:

#python
#Kopiera kod
#import namnetPåFilen
#Då sätts __name__ till filens modulnamn ("namnetPåFilen"),

#Och allt inuti if __name__ == '__main__': körs inte.

#Det är alltså ett sätt att skilja mellan:

#När filen körs som huvudprogram

#När filen används som modul / bibliotek i ett annat program

#Dvs: if __name__ == '__main__': kör koden bara när filen körs direkt, men inte när filen importeras som modul i en annan fil.



# Basic File Input/Output

CORPUS_PATH = 'C:\\Users\\tonny\\OneDrive\\Documents\\EDAN20\\EDAN20\\'

def count_letters(text, lc=True):
    letter_count = {}
    if lc:
        text = text.lower()
    for letter in text:
        if letter.lower() in alphabet:
            if letter in letter_count:
                letter_count[letter] += 1
            else:
                letter_count[letter] = 1
    return letter_count

try:
    # We open a file and we get a file object
    f_iliad = open(CORPUS_PATH + 'iliad.mb.txt', 'r', encoding='utf-8')
    iliad_txt = f_iliad.read()  # We read all the file
    f_iliad.close()  # We close the file
except:
    iliad_txt = None

iliad_stats = count_letters(iliad_txt)  # We count the letters
print(iliad_stats)

with open(CORPUS_PATH + 'iliad_stats.txt', 'w') as f:
    f.write(str(iliad_stats))
    # we automatically close the file

# This block writes the letter counts to a file and ensures the file is properly closed, without needing f.close() manually.

# Collecting a Corpus

classics_url = {'iliad': 'http://classics.mit.edu/Homer/iliad.mb.txt',
                'odyssey': 'http://classics.mit.edu/Homer/odyssey.mb.txt',
                'eclogue': 'http://classics.mit.edu/Virgil/eclogue.mb.txt',
                'georgics': 'http://classics.mit.edu/Virgil/georgics.mb.txt',
                'aeneid': 'http://classics.mit.edu/Virgil/aeneid.mb.txt'}
# Read the text from URLs

import requests

classics = {}
for key in classics_url:
    classics[key] = requests.get(classics_url[key]).text

# We remove the license information to keep only the text

import regex as re

for key in classics:
    classics[key] = re.search(r'^-+$(.+)^-+$',
                              classics[key],
                              re.M | re.S).group(1)

print(classics['iliad'][:50])

# Var tvungen och python -m pip install regex för att använda modulen

with open(CORPUS_PATH + 'iliad.txt', 'w') as f_il, open(CORPUS_PATH + 'odyssey.txt', 'w') as f_od:
    f_il.write(classics['iliad'])
    f_od.write(classics['odyssey'])


import json

with open(CORPUS_PATH + 'classics.json', 'w') as f:
    json.dump(classics, f)

with open(CORPUS_PATH + 'classics.json', 'r') as f:
    classics = json.loads(f.read())