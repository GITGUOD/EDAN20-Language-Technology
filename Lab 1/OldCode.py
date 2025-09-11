# Old code which have been used to solve the assignment. Unfortunately, not all code here is useful as they did not function as I thought. Could be useful though:
import regex as re
import math
import os

''' Old code, took too long
def id_if_forEachDocument(master_index, dir='Selma/'):
    tfidf = {}  # Dictionary for all documents

    files = get_files(dir, 'txt')
    all_words = list(master_index.keys())  # All unique words in corpus as each word is branded as a key while the position as value

    for file in files:
        tfidf[file] = {}
        for word in all_words:
            tfidf[file][word] = calculating_id_tf(word, master_index, file)
    return tfidf
'''

def tokenize(text, regex = r'\p{L}+'):
    match = re.finditer(regex, text) # search for all matches of a pattern in a string and return them as an iterator
    return match

#print(list(tokens)) # printar index och objekt

# Extracting indices

def text_to_idx(text):
    newIndex = dict()
    for match in text:
        word = match.group() # Returnerar strängar av matchen
        position = match.start() # returnerar positionen
        if word not in newIndex:
            newIndex[word] = []
        newIndex[word].append(position)
    return newIndex

# print(text_to_idx(tokens)) # printar  endast indexet på texten samt matchen


# Exercise 4? Reading one file
#Read one file, Mårbacka, marbacka.txt, set it in lowercase, tokenize it, and index it. Call this index idx

# Method for this:
def read_file_and_tokenize_and_index_it(file, regex = r'\p{L}+'):
    # Reading file
    text = open(file, 'r', encoding='utf-8').read().lower() # text = open(file).read().lower() # didn't work here as we needed the utf-8 for our swedish characters
    
    # Tokenize words (letters including Swedish characters)
    words = tokenize(text, regex)

    index = text_to_idx(words) # Index it

    return index

text = read_file_and_tokenize_and_index_it('Lab 1/Selma/marbacka.txt')
# print(text)


# Index "Mårbacka"
textMårbacka = read_file_and_tokenize_and_index_it(file = 'Lab 1/Selma/marbacka.txt', regex = 'mårbacka')
# print(textMårbacka)
# Stämmer med output på labben


# Reading the content of a folder

def get_files(dir = 'Lab 1/Selma/', suffix = 'txt'):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files

# print(get_files('Selma/', 'txt'))
 
# Creating a master index:

def readAllFilesTokenizeAndIndexAll(dir, suffix, regex):
    # Lägga till alla filer
    files = get_files(dir, suffix) # Den returnerar alla texter så som bannlyst.txt

    #Nu behöver vi kunna länka de hit så att metoden kan komma åt den via os klassen

    masterIndex = dict()
    for file in files:
        # OS metoden här hjälper oss öppna alla våra filer
        file_path = os.path.join(dir, file)
        index = read_file_and_tokenize_and_index_it(file_path, regex)

        for word, position in index.items():
            if word not in masterIndex:
                masterIndex[word] = dict()
            masterIndex[word][file] = position


    return masterIndex

master_Indexx = readAllFilesTokenizeAndIndexAll('Lab 1/Selma/', 'txt', r'\p{L}+')



# Old code, took to long and did not use the master index:

# Tf will be the relative frequency of the term in the document
def tfCalculations(word, file):
    # Öppna texten
    text = open(file, 'r', encoding='utf-8').read().lower() 
    # Tokenisera texten
    index = tokenize(text)
    tokens = []
    for match in index:
        tokens.append(match.group())
    frequencyOfTerm = tokens.count(word)
    totalNbrWords = len(tokens)

    result = frequencyOfTerm/totalNbrWords
    if(result < 0.0):
        return 0
    else:
        return result

# idf, the logarithm base 10 of the inverse document frequency.
def idfCalculations(word, master_index):
    total_docs = len(get_files('Lab 1/Selma/', 'txt'))
        
    if word not in master_index:
        return 0  # ordet finns inte i något dokument
    d_t = len(master_index[word])  # antal dokument där ordet finns
    return math.log10(total_docs / d_t)

# According to the wikipedia page: Term frequency–inverse document frequency slide
def calculating_id_tf(word, file, master_index):
    idf = idfCalculations(word, master_index)
    tf = tfCalculations(word, file)
    return tf * idf




# Creating a method to calc idif for the all documents, inte klar än
def idif_forAFewWordsEachDocument(master_index, dir='Lab 1/Selma/', listOfWords = ['gås', 'nils', 'känna', 'et']):
    tfidf = {}  # Dictionary for all documents

    files = get_files(dir, 'txt')
    #all_words = master_index.keys()  # All unique words in corpus as each word is branded as a key while the position as value

    for file in files:
        tfidf[file] = {}
        for word in listOfWords:
            text = os.path.join(dir, file)
            tfidf[file][word] = calculating_id_tf(word, text, master_index)
    return tfidf

print("hej", idif_forAFewWordsEachDocument(master_Indexx))
