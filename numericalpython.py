# Chapter 5: Python
# An introduction to linear algebra with NumPy and PyTorch by Pierre Nugues

# The Corpus

# We create a dictionary with URLs

classics_url = {'iliad': 'http://classics.mit.edu/Homer/iliad.mb.txt',
                'odyssey': 'http://classics.mit.edu/Homer/odyssey.mb.txt',
                'eclogue': 'http://classics.mit.edu/Virgil/eclogue.mb.txt',
                'georgics': 'http://classics.mit.edu/Virgil/georgics.mb.txt',
                'aeneid': 'http://classics.mit.edu/Virgil/aeneid.mb.txt'}

# Read URLs
import requests

classics = {}
for key in classics_url:
    classics[key] = requests.get(classics_url[key], verify=False).text

# We remove the license information to keep only the text

import regex as re

for key in classics:
    classics[key] = re.search(r'^-+$(.+)^-+$',
                              classics[key],
                              re.M | re.S).group(1)
    
classics['iliad'][:50]

# Write liad and Odyssey in two text files

PATH = 'C:\\Users\\tonny\\OneDrive\\Documents\\EDAN20\\EDAN20\\'

with open(PATH + 'iliad.txt', 'w') as f_il, open(PATH + 'odyssey.txt', 'w') as f_od:
    f_il.write(classics['iliad'])
    f_od.write(classics['odyssey'])

# STORE in a JSON file

import json
with open(PATH + 'classics.json', 'w') as f:
    json.dump(classics, f)

# Read it again

with open(PATH + 'classics.json', 'r') as f:
    classics = json.loads(f.read())

print("Finito")


# Utilities

alphabet = 'abcdefghijklmnopqrstuvwxyz'
class Text:
    """Text class to hold and process text"""

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(self, text: str = None):
        """The constructor called when an object
        is created"""

        self.content = text
        self.length = len(text)
        self.letter_counts = {}

    def count_letters(self, lc: bool = True) -> dict[str, int]:
        """Function to count the letters of a text"""

        letter_counts = {}
        if lc:
            text = self.content.lower()
        else:
            text = self.content
        for letter in text:
            if letter.lower() in self.alphabet:
                if letter in letter_counts:
                    letter_counts[letter] += 1
                else:
                    letter_counts[letter] = 1
        self.letter_counts = letter_counts
        return letter_counts

#Imports

import math
import random
import numpy as np
import torch # pip install torch
random.seed(4321)
np.random.seed(4321)
torch.manual_seed(4321)

# Let us read Homer's Iliad and Odyssey and Virgil's Eclogue, Georgics, and Aeneid.
titles = list(classics.keys())
print(titles)

texts = []
for title in titles:
    texts += [classics[title]]
cnt_dicts = []
for text in texts:
    cnt_dicts += [Text(text).count_letters()]
cnt_lists = []
for cnt_dict in cnt_dicts:
    cnt_lists += [list(map(lambda x: cnt_dict.get(x, 0),
                           alphabet))]
    
print(cnt_lists[0][:3])

for i, cnt_list in enumerate(cnt_lists):
    print(titles[i], cnt_lists[i][:10])

# Vectors

# NumPy #Bibliotek för att skapa matriser som är snabbare och effektivare än vanliga python listor. Man slipper loopa och kan göra matematiska operationer

np.array([2, 3]) # 1D matris [2,3]
np.array([1, 2, 3]) # 1D matris [1,2,3]

# Vectors of letter counts

iliad_cnt = np.array(cnt_lists[0]) # Skapar arrays med antalet bokstäver
odyssey_cnt = np.array(cnt_lists[1])
eclogue_cnt = np.array(cnt_lists[2])
georgics_cnt = np.array(cnt_lists[3])
aeneid_cnt = np.array(cnt_lists[4])

print(iliad_cnt)

print(odyssey_cnt)

#The datatype

print(odyssey_cnt.dtype)

vector = np.array([1, 2, 3], dtype='int32')
print(vector)

print(vector.dtype) # vilken datatyp arrayen har

vector = np.array([1, 2, 3], dtype='float64') #datatyp float med 64bitar
print(vector)

print(np.array([0, 1, 2, 3], dtype='bool'))

#The vector size

print(odyssey_cnt.shape) # Printar ut storleken på vektorn på odyssey

#Indices and Slices

vector = np.array([1, 2, 3, 4])
vector[1]   # 2
vector[:1]  # array([1])
vector[1:3]  # array([2, 3])

#Operations

np.array([1, 2, 3]) + np.array([4, 5, 6]) # output: array([5, 7, 9])

3 * np.array([1, 2, 3]) # ger oss outputen: array([3, 6, 9])

iliad_cnt + odyssey_cnt      # array([88643,  15533,  20138, ...])

iliad_cnt - odyssey_cnt      # array([13389,  2343,  2978, ...])

iliad_cnt - 2 * odyssey_cnt  # array([-24238,  -4252,  ...])

# Comparison with lists

[1, 2, 3] + [4, 5, 6] # output: [1, 2, 3, 4, 5, 6]
3 * [1, 2, 3] # output: [3, 6, 9]


# PyTorch 
    # PyTorch är ett bibliotek i Python för maskininlärning och djupinlärning.
        # Det används mycket för neurala nätverk, men innehåller också matematiska verktyg som liknar NumPy.
# Tensors

torch.tensor([2, 3])
torch.tensor([1, 2, 3])

# Output: tensor([1, 2, 3])
iliad_cnt_pt = torch.tensor(cnt_lists[0])

print(iliad_cnt_pt) # Output: tensor([51016,  8938, 11558, 28331, 77461, 16114, 12595, 50192, 38149,  1624....])
# En tensor är PyTorchs version av en NumPy-array.

# Types 

torch.tensor([1, 2, 3]).dtype # Output: torch.int64

torch.tensor([1, 2, 3], dtype=torch.float64) # tensor([1., 2., 3.], dtype=torch.float64)
 
torch.tensor([0, 1, 2, 3], dtype=torch.bool) # tensor([False,  True,  True,  True])

# Size

iliad_cnt_pt.size()
# Output: torch.Size([26])

# NumPy / PyTorch Conversion
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array) #Omvandla NumPy-array till PyTorch-tensor
print(tensor) # Output: tensor([1, 2, 3])

tensor = torch.tensor([1, 2, 3])
np_array = tensor.numpy()
print(np_array) # Output: array([1, 2, 3])

# Device

torch.cuda.is_available() #Output: False

torch.backends.mps.is_available() #Output: True

torch.device('cpu') #Output: device(type='cpu')

torch.device('mps') #Output: device(type='mps')

tensor = torch.tensor([1, 2, 3])
print(tensor.device) # Output: device(type='cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(device) # Output: device(type='mps')

tensor = torch.tensor([1, 2, 3], device=device)
print(tensor) # Output: tensor([1, 2, 3], device='mps:0')

