# Lab 1, we will create an indexer, index consists of rows with one word per row and
# the list of files and positions where this words occur. Such a row is called a posting list.
import math
import os
import pickle
import regex as re

# We have downloaded Selmas file

'''
Write a program that reads one document file_name.txt and outputs an index file: file_name.idx:

The index file will contain all the unique words in the document, where each word is associated with the list of its positions in the document.
You will represent this index as a dictionary, where the keys will be the words, and the values, the lists of positions
As words, you will consider all the strings of letters that you will set in lower case. You will not index the rest (i.e. numbers, punctuations, or symbols).
To extract the words, use Unicode regular expressions. Do not use \\w+, for instance, but the Unicode equivalent.
The word positions will correspond to the number of characters from the beginning of the file. (The word offset from the beginning)
You will use the finditer() method to find the positions of the words. This will return you match objects, where you will get the matches and the positions with the group() and start() methods.
You will use the pickle package to write your dictionary in an file, see https://wiki.python.org/moin/UsingPickle.
'''

#ToDo:

# Make the program so that it can open and read one document file: Finished

text = open('Lab 1/Selma/bannlyst.txt').read()

# 1. Index file will contain unique words. Varje ord kommer ha en lista med positioner i dokumentet så som föreläsningen med index antar jag typ (11, 34).
# Dictionary ska användas där nycklar kommer att vara ord och värdena ska vara positioner vilket gör att orden blir unika.
# ord kommer vara i lowercase. Endast ord i listan och inte tecken, symboler eller punkter etc
# När vi extractar orden, använd unicode regular expressions (regex) ingen \w+.
# The word positions will correspond to the number of characters from the beginning of the file. (The word offset from the beginning) 
# finditer() kommer att användas för att hitta positioner, matcha objekt, where you will get the matches and the positions with the group() and start() methods.

# ALLT ovan är fixat


index = dict() # Skapa en dictionary
wholeText = re.finditer(r'\p{L}+', text)
for match in wholeText: # För varje match i texten, L för letters då vi inte vill ha andra symboler etc. Lägga till + för att få med hela ordet
    word = match.group().lower() # Varje match kör vi till lower case // group() metoden returnerar en del av regexet, dvs en sträng
    position = match.start() # hämtar positionen av ordet
    if word not in index: # Lägga till ordet i vår dictionary för första gången
        index[word] = []
    index[word].append(position) # Appenda positionen i listan med vår nyckel

pickle.dump(index, open("index.p", "wb"))

#Output index file via pickle
# Klart!


# Öppna filen igen:

text = pickle.load(open("index.p", "rb"))

regex = r'\p{L}+' # Kan vara vad som egentligen, t.ex "b.c" om du vill matcha med back-kajsa (början)
# Detta funkar eftersom vi skippar commat, och binde streck vilket gör att Back och Kajsa blir olika ord.

line = 'En gång hade de på Mårbacka en barnpiga, som hette Back-Kajsa'

match = re.findall(regex, line)

# print(index) #printar texten
# print(match) #printar line

# Using regex, write tokenize(text) function to tokenize a text. Return their positions.

def tokenize(text, regex = r'\p{L}+'):
    match = re.finditer(regex, text)
    return match

tokens = tokenize(line)
#print(list(tokens)) # printar index och objekt

# Extracting indices

def text_to_idx(text):
    newIndex = dict()
    for match in text:
        word = match.group()
        position = match.start()
        if word not in newIndex:
            newIndex[word] = []
        newIndex[word].append(position)
    return newIndex

tokens = tokenize(line.lower().strip())
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
    

#print(readAllFilesTokenizeAndIndexAll('Selma/', 'txt', r'\bsamlar\b')) # Skrev innan 'samlar' med den hämtade också t.ex 'samlar'

#print(readAllFilesTokenizeAndIndexAll('Selma/', 'txt', r'\bmårbacka\b')) # 




# Concordances
# Write a concordance(word, master_index, window) function to extract the concordances of a word within a window of window characters

def concordance(word, master_index, window):

    # Extracting a word within a window of characters
    try:
        
        for filename, position in master_index[word.lower()].items():
            print(filename)

            # Vi behöver hämta texterna för att kunna sedan hitta snippet/meningen vid ordet
            file_path = os.path.join('Selma/', filename)
            text = open(file_path, 'r', encoding='utf-8').read().lower()

            # iterera till positionen av ordet vi letar efter och klipper ett snipp
            for pos in position:
                start = max(0, pos - window)
                end = min(len(text), pos + window)
                snippet = text[start:end].replace('\n', ' ')
                print("\t" + snippet)
        return 'End'


    except Exception as e:
        return f"The word '{word}' does not exist in master_index: {e}"

masterIndex = readAllFilesTokenizeAndIndexAll('Lab 1/Selma/', 'txt', r'\p{L}+')

concordance('samlar', masterIndex, 25)

# Calculating TF-IDF

"""Tf will be the relative frequency of the term in the document and
idf, the logarithm base 10 of the inverse document frequency. """

# Tf will be the relative frequency of the term in the document
def tfCalculations(word, master_index, filename):

    word_count = []

    frequencyOfTerm = 0
    for w, positions in master_index.items():
        if filename in positions:  # kolla att ordet finns i dokumentet
            for _ in range(len(positions[filename])):
                word_count.append(w)

    frequencyOfTerm = word_count.count(word)
    totalNbrWords = len(word_count)

    if totalNbrWords == 0:   # Zero Division
        return 0

    result = frequencyOfTerm/totalNbrWords
    if result <= 0.0 or frequencyOfTerm <= 0.0:
        return 0
    else:
        return result


    
# idf, the logarithm base 10 of the inverse document frequency.
def idfCalculations(word, master_index):
    total_docs = len(get_files('Lab 1/Selma/', 'txt'))
        
    if word not in master_index:
        return 0  # ordet finns inte i något dokument
    
    d_t = len(master_index[word])  # # antal dokument där ordet finns

    if d_t == 0:
        return 0
    
    return math.log10(total_docs / d_t)

# According to the wikipedia page: Term frequency–inverse document frequency slide
def calculating_id_tf(word, master_index, filename):
    idf = idfCalculations(word, master_index)
    tf = tfCalculations(word, master_index, filename)
    return tf * idf
print('iftf calc')
print(calculating_id_tf('nils', masterIndex, 'jerusalem.txt'))

# Fixat ovan!


master_Indexx = readAllFilesTokenizeAndIndexAll('Lab 1/Selma/', 'txt', r'\p{L}+')
#print(master_Indexx.keys())


result = calculating_id_tf(word='nils', master_index=master_Indexx, filename='marbacka.txt')
resul2 = calculating_id_tf(word='et', master_index=master_Indexx, filename='marbacka.txt')
resul3 = calculating_id_tf(word='nils', master_index=master_Indexx, filename='jerusalem.txt')


print('idtf results: nils', result)
print('idtf results et:', resul2)
print('correct, now to the jerusalem.txt, nils', resul3)


# Creating a method to calc idif for the all documents, inte klar än
def idif_forAFewWordsEachDocument(master_index, dir='Lab 1/Selma/', listOfWords = ['gås', 'nils', 'känna', 'et']):
    tfidf = {}  # Dictionary for all documents

    files = get_files(dir, 'txt')
    #all_words = master_index.keys()  # All unique words in corpus as each word is branded as a key while the position as value

    for file in files:
        tfidf[file] = {}
        for word in listOfWords:
            text = os.path.join(dir, file)
            tfidf[file][word] = calculating_id_tf(word, master_index, text)
    return tfidf

#print(idif_forAFewWordsEachDocument(master_Indexx))


def count_words_per_file(master_index, files):
    total_words_per_file = {}
    for file in files:
        total_count = 0
        for w in master_index:
            if file in master_index[w]:
                total_count += len(master_index[w][file])
        total_words_per_file[file] = total_count
    return total_words_per_file


# Creating a method to calculate TF-IDF for all documents
def id_if_forEachDocument(master_index, dir='Lab 1/Selma/'):
    tfidf = {}
    files = get_files(dir, 'txt')
    all_words = list(master_index.keys())

    # Räknar totala ord per dokument
    total_words_per_file = count_words_per_file(master_index, files)

    # Calculate TF-IDF
    for file in files:
        tfidf[file] = {}
        for word in all_words:
            # TF, term freq: basically how often the term comes up in a document
            tf = 0
            if word in master_index and file in master_index[word]:
                frequency = len(master_index[word][file])
                total_words = total_words_per_file[file]
                if total_words > 0:
                    tf = frequency / total_words

            # IDF, inversion document freq: basically how unique a word is across all elements/documents
            idf = 0
            if word in master_index:
                doc_count = len(files)
                docs_with_word = len(master_index[word])
                if docs_with_word > 0:
                    idf = math.log10(doc_count / docs_with_word)

            # TF-IDF calculates how unique one element is in one document incomparison to its counterparts in other documents, for instance:
            # 'the' has a low tf_idf score as it comes up very often in across all documents.
            tfidf[file][word] = tf * idf

    return tfidf



print('idif for the whole dictionary for each file')
print(id_if_forEachDocument(master_Indexx))

# Comparing Documents
    # Cosine similarity
# According to lecture chapter 9 for the cosine is the dot product D1*D"/ Absolute value of D1 multiplied by absolut value of D2
# The theory is to calculate how similar both text are. The values are between 0 and 1.

def cosine_similarity(document1, document2):
    dot_product = 0
    D1 = 0
    D2 = 0

    for word in document1:
        tfidf1 = document1[word]
        tfidf2 = document2.get(word, 0)
        dot_product += tfidf1 * tfidf2
        D1 += tfidf1 ** 2
        D2 += tfidf2 ** 2

    D1 = math.sqrt(D1)
    D2 = math.sqrt(D2)

    if D1 == 0 or D2 == 0:
        return 0

    return dot_product / (D1 * D2)

# Matrix over the most similar text based on the cosine similarity calculations
def similarityMatrix():
    files = get_files()
    tfidf_forAllDocuments = id_if_forEachDocument(master_Indexx)

    similarity_matrix = {}
    most_sim_doc1 = ""
    most_sim_doc2 = ""
    max_similarity = 0

    for file1 in files:
        similarity_matrix[file1] = {}
        for file2 in files:
             # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_forAllDocuments[file1], tfidf_forAllDocuments[file2])
            similarity_matrix[file1][file2] = similarity

            # Update most similar pair 
            if file1 != file2:
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_sim_doc1 = file1
                    most_sim_doc2 = file2
    
    print("Most similar:", most_sim_doc1, most_sim_doc2, "Similarity:", max_similarity)
    return similarity_matrix, most_sim_doc1, most_sim_doc2, max_similarity

similarityMatrix()





