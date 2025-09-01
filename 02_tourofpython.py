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

# Slices
alphabet = 'abcdefghijklmnopqrstuvwxyz'
first = alphabet[0:3]
second = alphabet[:3] #Samma som first
third = alphabet[3:6] # index 3 till 5 så: 'def'
fourth = alphabet[-3:] # 'xyz'

print(first)  # 'abc'
print(second) # 'abc'
print(third)  # 'def'
print(fourth) # 'xyz'