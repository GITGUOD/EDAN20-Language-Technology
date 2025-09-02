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

'''
alphabet = 'abcdefghijklmnopqrstuvwxyz'
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
