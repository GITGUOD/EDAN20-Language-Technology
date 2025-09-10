# Chapter 5: Python
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

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

# Device - var PyTorch kör beräkningarna — CPU, GPU eller annan accelerator.

torch.cuda.is_available() #Output: False

torch.backends.mps.is_available() #Output: True

torch.device('cpu') #Output: device(type='cpu')

torch.device('mps') #Output: device(type='mps')

'''
cuda → NVIDIA GPU (vanlig på Windows/Linux med NVIDIA-kort)

mps → Apple Silicon GPU (M1/M2 Mac)

'''

tensor = torch.tensor([1, 2, 3])
print(tensor.device) # Output: device(type='cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

'''
torch.device('cpu')  # CPU
torch.device('mps')  # Apple GPU
torch.device säger var tensor ska lagras och beräkningar ska ske.

CPU = vanlig processor, MPS/CUDA = GPU.
'''

print(device) # Output: device(type='mps')

tensor = torch.tensor([1, 2, 3], device=device)
print(tensor) # Output: tensor([1, 2, 3], device='mps:0') Vektor, välja enhet

tensor = torch.tensor([1, 2, 3])
tensor.to(device) # Output: tensor([1, 2, 3], device='mps:0')


# NumPy Functions


np.set_printoptions(precision=3)
np.sqrt(iliad_cnt) # Printar ut roten av alla element, Tar kvadratroten av varje element i arrayen.

np.cos(iliad_cnt)

# math.sqrt(iliad_cnt) Ger oss error  math.sqrt(x) fungerar bara på ett enda tal (float eller int).

np_sqrt = np.vectorize(math.sqrt)
np_sqrt(iliad_cnt) # Output: array([225.867,  94.541, 107.508, 168.318, 278.318, 126.941, 112.227, 224.036, 195.318,  40.299,  66.43 , 159.082, 129.019, 205.409, 226.427,  95.415,  16.823, 190.929, 203.079, 232.751, 135.672,  77.846, 125.16 ,  24.434, 109.115,  16.852])

np.sum(odyssey_cnt) # Output:  472937 # Summor av alla element i arrayen, finns motsvarande i pyTorch under

iliad_dist = iliad_cnt / np.sum(iliad_cnt)
odyssey_dist = odyssey_cnt / np.sum(odyssey_cnt)

print(iliad_dist)
#Output:
'''
array([0.081, 0.014, 0.018, 0.045, 0.123, 0.026, 0.02 , 0.08 , 0.061,
       0.003, 0.007, 0.04 , 0.026, 0.067, 0.081, 0.014, 0.   , 0.058,
       0.065, 0.086, 0.029, 0.01 , 0.025, 0.001, 0.019, 0.   ])
'''

print(odyssey_dist)

'''
array([0.08 , 0.014, 0.018, 0.044, 0.126, 0.022, 0.021, 0.074, 0.061,
       0.001, 0.008, 0.04 , 0.028, 0.067, 0.082, 0.014, 0.001, 0.054,
       0.066, 0.086, 0.033, 0.01 , 0.027, 0.001, 0.023, 0.   ])
'''

# PyTorch
torch.sqrt(iliad_cnt_pt) # Tar kvadratroten av varje element i arrayen. Samma funktionalitet som NumPy.

#Output:
'''
tensor([225.8672,  94.5410, 107.5081, 168.3181, 278.3182, 126.9409, 112.2274,
        224.0357, 195.3177,  40.2989,  66.4304, 159.0817, 129.0194, 205.4093,
        226.4266,  95.4149,  16.8226, 190.9293, 203.0788, 232.7509, 135.6724,
         77.8460, 125.1599,  24.4336, 109.1146,  16.8523])
'''
torch.sum(iliad_cnt_pt)

#Output: tensor(629980)

#Dot Product

np.dot(iliad_dist, odyssey_dist) # Output: 0.06581149298284382

torch.dot(torch.tensor(iliad_dist), torch.tensor(odyssey_dist)) # Output: tensor(0.0658, dtype=torch.float64)

# Dot product (prickprodukt) = summan av produkter mellan motsvarande element, Ger ett mått på likhet mellan två frekvensvektorer.

# Norm
  #Längd 

np.linalg.norm([1.0, 2.0, 3.0]) # Output: 3.7416573867739413 eftersom det under:
#Längd √(1² + 2² + 3²) då vi normaliserar de

torch.linalg.vector_norm(torch.tensor([1.0, 2.0, 3.0])) # Output: tensor(3.7417)

# Cosine

(iliad_dist @ odyssey_dist) / (
    np.linalg.norm(iliad_dist) *
    np.linalg.norm(odyssey_dist))

# Output: 0.999078711....

#  Delar med produkten av normerna → får cosinus mellan vektorerna

# Matrices

hv_cnts = np.array(cnt_lists)
print(hv_cnts)

# Output av våran matris av listor av listor
'''
array([[51016,  8938, 11558, 28331, 77461, 16114, 12595, 50192, 38149,
         1624,  4413, 25307, 16646, 42193, 51269,  9104,   283, 36454,
        41241, 54173, 18407,  6060, 15665,   597, 11906,   284],
       [37627,  6595,  8580, 20736, 59777, 10449,  9803, 34785, 28793,
          424,  3631, 18948, 13058, 31888, 38776,  6679,   256, 25665,
        31348, 40479, 15404,  4803, 12989,   350, 10970,   124],
       [ 2716,   577,   722,  1440,  4363,   846,   806,  2508,  2250,
           22,   268,  1807,  1043,  2248,  2947,   569,    12,  2235,
         2617,  2939,  1030,   360,  1023,    38,   905,    22],
       [ 6841,  1618,  2016,  4027, 12110,  2424,  2147,  6987,  6035,
           59,   782,  4308,  2027,  6552,  6957,  1669,    53,  6702,
         7142,  8712,  2583,   902,  2480,    85,  1457,    64],
       [36675,  6867, 10023, 23862, 55367, 11618,  9606, 33055, 30576,
          907,  2702, 18766, 10201, 32254, 32594,  8343,   530, 32074,
        36429, 39478, 13714,  4374, 11097,   560,  8053,   270]])
'''

# The size

hv_cnts.shape # Ger oss (5, 26)

# The data type:
hv_cnts.dtype #Output: dtype('int64')

#Indices and Slices

iliad_cnt[2] # Output 11558
hv_cnts[1, 2] # Output: 8580

# Slices

hv_cnts[:, 2]
# Output: array([11558,  8580,   722,  2016, 10023])
hv_cnts[1, :]

# Output
'''
array([37627,  6595,  8580, 20736, 59777, 10449,  9803, 34785, 28793,
         424,  3631, 18948, 13058, 31888, 38776,  6679,   256, 25665,
       31348, 40479, 15404,  4803, 12989,   350, 10970,   124])
'''
hv_cnts[1, :2]
# Output: array([37627,  6595])
hv_cnts[3, 2:4]
# Output: array([2016, 4027])
hv_cnts[3:, 2:4]
'''
array([[ 2016,  4027],
       [10023, 23862]])
'''

# Number of indices
odyssey_cnt.ndim
# Output: 1
hv_cnts.ndim
# Output: 2

#Addition and multiplication by a scalar
# hv_cnts - 2 * hv_cnts

'''
array([[-51016,  -8938, -11558, -28331, -77461, -16114, -12595, -50192,
        -38149,  -1624,  -4413, -25307, -16646, -42193, -51269,  -9104,
          -283, -36454, -41241, -54173, -18407,  -6060, -15665,   -597,
        -11906,   -284],
       [-37627,  -6595,  -8580, -20736, -59777, -10449,  -9803, -34785,
        -28793,   -424,  -3631, -18948, -13058, -31888, -38776,  -6679,
          -256, -25665, -31348, -40479, -15404,  -4803, -12989,   -350,
        -10970,   -124],
       [ -2716,   -577,   -722,  -1440,  -4363,   -846,   -806,  -2508,
         -2250,    -22,   -268,  -1807,  -1043,  -2248,  -2947,   -569,
           -12,  -2235,  -2617,  -2939,  -1030,   -360,  -1023,    -38,
          -905,    -22],
       [ -6841,  -1618,  -2016,  -4027, -12110,  -2424,  -2147,  -6987,
         -6035,    -59,   -782,  -4308,  -2027,  -6552,  -6957,  -1669,
           -53,  -6702,  -7142,  -8712,  -2583,   -902,  -2480,    -85,
         -1457,    -64],
       [-36675,  -6867, -10023, -23862, -55367, -11618,  -9606, -33055,
        -30576,   -907,  -2702, -18766, -10201, -32254, -32594,  -8343,
          -530, -32074, -36429, -39478, -13714,  -4374, -11097,   -560,
         -8053,   -270]])
'''

# PyTorch
hv_cnts_pt = torch.tensor(cnt_lists)
print(hv_cnts_pt)

#Output:
'''
tensor([[51016,  8938, 11558, 28331, 77461, 16114, 12595, 50192, 38149,  1624,
          4413, 25307, 16646, 42193, 51269,  9104,   283, 36454, 41241, 54173,
         18407,  6060, 15665,   597, 11906,   284],
        [37627,  6595,  8580, 20736, 59777, 10449,  9803, 34785, 28793,   424,
          3631, 18948, 13058, 31888, 38776,  6679,   256, 25665, 31348, 40479,
         15404,  4803, 12989,   350, 10970,   124],
        [ 2716,   577,   722,  1440,  4363,   846,   806,  2508,  2250,    22,
           268,  1807,  1043,  2248,  2947,   569,    12,  2235,  2617,  2939,
          1030,   360,  1023,    38,   905,    22],
        [ 6841,  1618,  2016,  4027, 12110,  2424,  2147,  6987,  6035,    59,
           782,  4308,  2027,  6552,  6957,  1669,    53,  6702,  7142,  8712,
          2583,   902,  2480,    85,  1457,    64],
        [36675,  6867, 10023, 23862, 55367, 11618,  9606, 33055, 30576,   907,
          2702, 18766, 10201, 32254, 32594,  8343,   530, 32074, 36429, 39478,
         13714,  4374, 11097,   560,  8053,   270]])
'''

hv_cnts_pt.dtype # Type: Output: torch.int64

hv_cnts_pt.dim() # Output: 2
hv_cnts_pt.size() # Output: torch.Size([5, 26])

hv_cnts_pt.size(dim=0)   # 5
hv_cnts_pt.size(dim=1)   # 26

hv_cnts_pt.size(dim=-1)  # 26
hv_cnts_pt.size(dim=-2)  # 5

# NumPy Functions
np.set_printoptions(precision=3)
np.cos(hv_cnts)

#Output:
'''
array([[-0.948, -0.986, -0.997,  0.993, -0.315, -0.717, -0.938, -0.338,
        -0.802, -0.979, -0.592, -0.099, -0.268,  0.159, -0.22 ,  0.944,
         0.967,  0.505, -0.255,  0.812, -0.918, -0.991,  0.524,  0.995,
         0.804,  0.309],
       [-0.99 , -0.699, -0.952,  0.082,  0.339,  0.998,  0.333,  0.281,
        -0.954, -0.993,  0.777, -0.493,  0.03 ,  0.671, -0.779,  1.   ,
        -0.04 , -0.239,  0.373, -0.913, -0.717, -0.88 , -0.085, -0.284,
         0.904, -0.093],
       [-0.093,  0.495,  0.844,  0.408, -0.782, -0.613, -0.18 ,  0.533,
         0.814, -1.   , -0.57 , -0.834,  1.   ,  0.189,  0.983, -0.932,
         0.844, -0.241, -0.999,  0.04 ,  0.904, -0.284,  0.4  ,  0.955,
         0.976, -1.   ],
       [ 0.181, -0.997,  0.62 ,  0.867, -0.668,  0.258, -0.275,  0.995,
        -1.   , -0.771, -0.967, -0.64 , -0.782,  0.207,  0.057, -0.686,
        -0.918, -0.555, -0.399, -0.935,  0.819, -0.935, -0.283, -0.984,
         0.765,  0.392],
       [ 0.999,  0.867,  0.249,  0.033,  0.909,  0.925,  0.548,  0.669,
        -0.435, -0.606,  0.974, -0.299, -0.969, -0.744, -1.   ,  0.48 ,
        -0.599, -0.09 ,  0.615,  0.734, -0.583,  0.619,  0.626,  0.699,
        -0.455,  0.984]])
'''

np.sum(hv_cnts) # Output: 1705964

np.sum(hv_cnts, axis=0) # OUTPUT:

'''
array([134875,  24595,  32899,  78396, 209078,  41451,  34957, 127527,
       105803,   3036,  11796,  69136,  42975, 115135, 132543,  26364,
         1134, 103130, 118777, 145781,  51138,  16499,  43254,   1630,
        33291,    764])
'''

np.sum(hv_cnts, axis=1)
#Output: array([629980, 472937,  36313,  96739, 469995])

#PyTorch

torch.sum(hv_cnts_pt) # Output: tensor(1705964)

torch.sum(hv_cnts_pt, dim=0)  # array([ 134885,  24605,  32901, ...])

'''
tensor([134875,  24595,  32899,  78396, 209078,  41451,  34957, 127527, 105803,
          3036,  11796,  69136,  42975, 115135, 132543,  26364,   1134, 103130,
        118777, 145781,  51138,  16499,  43254,   1630,  33291,    764])
'''

torch.sum(hv_cnts_pt, dim=1)  # array([630019, 472978,  36332,  96758, 470034])

'''
tensor([629980, 472937,  36313,  96739, 469995])

'''

# Transposing and Reshaping Arrays

hv_cnts.T
#Output
'''
array([[51016, 37627,  2716,  6841, 36675],
       [ 8938,  6595,   577,  1618,  6867],
       [11558,  8580,   722,  2016, 10023],
       [28331, 20736,  1440,  4027, 23862],
       [77461, 59777,  4363, 12110, 55367],
       [16114, 10449,   846,  2424, 11618],
       [12595,  9803,   806,  2147,  9606],
       [50192, 34785,  2508,  6987, 33055],
       [38149, 28793,  2250,  6035, 30576],
       [ 1624,   424,    22,    59,   907],
       [ 4413,  3631,   268,   782,  2702],
       [25307, 18948,  1807,  4308, 18766],
       [16646, 13058,  1043,  2027, 10201],
       [42193, 31888,  2248,  6552, 32254],
       [51269, 38776,  2947,  6957, 32594],
       [ 9104,  6679,   569,  1669,  8343],
       [  283,   256,    12,    53,   530],
       [36454, 25665,  2235,  6702, 32074],
       [41241, 31348,  2617,  7142, 36429],
       [54173, 40479,  2939,  8712, 39478],
       [18407, 15404,  1030,  2583, 13714],
       [ 6060,  4803,   360,   902,  4374],
       [15665, 12989,  1023,  2480, 11097],
       [  597,   350,    38,    85,   560],
       [11906, 10970,   905,  1457,  8053],
       [  284,   124,    22,    64,   270]])
'''
iliad_cnt.T # Output:
'''
array([51016,  8938, 11558, 28331, 77461, 16114, 12595, 50192, 38149,
        1624,  4413, 25307, 16646, 42193, 51269,  9104,   283, 36454,
       41241, 54173, 18407,  6060, 15665,   597, 11906,   284])
'''
np.array([iliad_cnt])
'''
array([[51016,  8938, 11558, 28331, 77461, 16114, 12595, 50192, 38149,
         1624,  4413, 25307, 16646, 42193, 51269,  9104,   283, 36454,
        41241, 54173, 18407,  6060, 15665,   597, 11906,   284]])
    
'''

np.array([iliad_cnt]).shape # (1, 26)

np.array([iliad_cnt]).T


# Output:
'''
array([[51016],
       [ 8938],
       [11558],
       [28331],
       [77461],
       [16114],
       [12595],
       [50192],
       [38149],
       [ 1624],
       [ 4413],
       [25307],
       [16646],
       [42193],
       [51269],
       [ 9104],
       [  283],
       [36454],
       [41241],
       [54173],
       [18407],
       [ 6060],
       [15665],
       [  597],
       [11906],
       [  284]])
'''

iliad_cnt.reshape(1, -1) #Output:
'''
array([[51016,  8938, 11558, 28331, 77461, 16114, 12595, 50192, 38149,
         1624,  4413, 25307, 16646, 42193, 51269,  9104,   283, 36454,
        41241, 54173, 18407,  6060, 15665,   597, 11906,   284]])
'''

iliad_cnt.reshape(-1, 1)

# Output:
'''
array([[51016],
       [ 8938],
       [11558],
       [28331],
       [77461],
       [16114],
       [12595],
       [50192],
       [38149],
       [ 1624],
       [ 4413],
       [25307],
       [16646],
       [42193],
       [51269],
       [ 9104],
       [  283],
       [36454],
       [41241],
       [54173],
       [18407],
       [ 6060],
       [15665],
       [  597],
       [11906],
       [  284]])
'''

# Reshapar på olika sätt beroende på hur du vill ha det.

torch.unsqueeze(torch.tensor([1, 2, 3]), 1)
#Output:
'''
tensor([[1],
        [2],
        [3]])
'''

# Broadcasting

# Relative frequencies of the letter counts

iliad_dist = (1/np.sum(iliad_cnt)) * iliad_cnt
odyssey_dist = (1/np.sum(odyssey_cnt)) * odyssey_cnt
iliad_cnt / np.sum(iliad_cnt)

'''
Output:

array([0.081, 0.014, 0.018, 0.045, 0.123, 0.026, 0.02 , 0.08 , 0.061,
       0.003, 0.007, 0.04 , 0.026, 0.067, 0.081, 0.014, 0.   , 0.058,
       0.065, 0.086, 0.029, 0.01 , 0.025, 0.001, 0.019, 0.   ])
'''

odyssey_cnt / np.sum(odyssey_cnt)

'''
Output:
array([0.08 , 0.014, 0.018, 0.044, 0.126, 0.022, 0.021, 0.074, 0.061,
       0.001, 0.008, 0.04 , 0.028, 0.067, 0.082, 0.014, 0.001, 0.054,
       0.066, 0.086, 0.033, 0.01 , 0.027, 0.001, 0.023, 0.   ])
'''

# We can apply an elementwise multiplication or division

np.array([np.sum(hv_cnts, axis=1)]).T

'''
Output:
array([[629980],
       [472937],
       [ 36313],
       [ 96739],
       [469995]])
'''

hv_dist = hv_cnts / np.array([np.sum(hv_cnts, axis=1)]).T
print(hv_dist)

'''
Output:
array([[0.081, 0.014, 0.018, 0.045, 0.123, 0.026, 0.02 , 0.08 , 0.061,
        0.003, 0.007, 0.04 , 0.026, 0.067, 0.081, 0.014, 0.   , 0.058,
        0.065, 0.086, 0.029, 0.01 , 0.025, 0.001, 0.019, 0.   ],
       [0.08 , 0.014, 0.018, 0.044, 0.126, 0.022, 0.021, 0.074, 0.061,
        0.001, 0.008, 0.04 , 0.028, 0.067, 0.082, 0.014, 0.001, 0.054,
        0.066, 0.086, 0.033, 0.01 , 0.027, 0.001, 0.023, 0.   ],
       [0.075, 0.016, 0.02 , 0.04 , 0.12 , 0.023, 0.022, 0.069, 0.062,
        0.001, 0.007, 0.05 , 0.029, 0.062, 0.081, 0.016, 0.   , 0.062,
        0.072, 0.081, 0.028, 0.01 , 0.028, 0.001, 0.025, 0.001],
       [0.071, 0.017, 0.021, 0.042, 0.125, 0.025, 0.022, 0.072, 0.062,
        0.001, 0.008, 0.045, 0.021, 0.068, 0.072, 0.017, 0.001, 0.069,
        0.074, 0.09 , 0.027, 0.009, 0.026, 0.001, 0.015, 0.001],
       [0.078, 0.015, 0.021, 0.051, 0.118, 0.025, 0.02 , 0.07 , 0.065,
        0.002, 0.006, 0.04 , 0.022, 0.069, 0.069, 0.018, 0.001, 0.068,
        0.078, 0.084, 0.029, 0.009, 0.024, 0.001, 0.017, 0.001]])
'''

# The Hadamard product

hv_dist * hv_dist

'''
Output:
array([[6.558e-03, 2.013e-04, 3.366e-04, 2.022e-03, 1.512e-02, 6.543e-04,
        3.997e-04, 6.348e-03, 3.667e-03, 6.645e-06, 4.907e-05, 1.614e-03,
        6.982e-04, 4.486e-03, 6.623e-03, 2.088e-04, 2.018e-07, 3.348e-03,
        4.286e-03, 7.395e-03, 8.537e-04, 9.253e-05, 6.183e-04, 8.980e-07,
        3.572e-04, 2.032e-07],
       [6.330e-03, 1.945e-04, 3.291e-04, 1.922e-03, 1.598e-02, 4.881e-04,
        4.296e-04, 5.410e-03, 3.707e-03, 8.038e-07, 5.894e-05, 1.605e-03,
        7.623e-04, 4.546e-03, 6.722e-03, 1.994e-04, 2.930e-07, 2.945e-03,
        4.394e-03, 7.326e-03, 1.061e-03, 1.031e-04, 7.543e-04, 5.477e-07,
        5.380e-04, 6.874e-08],
       [5.594e-03, 2.525e-04, 3.953e-04, 1.573e-03, 1.444e-02, 5.428e-04,
        4.927e-04, 4.770e-03, 3.839e-03, 3.670e-07, 5.447e-05, 2.476e-03,
        8.250e-04, 3.832e-03, 6.586e-03, 2.455e-04, 1.092e-07, 3.788e-03,
        5.194e-03, 6.551e-03, 8.045e-04, 9.828e-05, 7.936e-04, 1.095e-06,
        6.211e-04, 3.670e-07],
       [5.001e-03, 2.797e-04, 4.343e-04, 1.733e-03, 1.567e-02, 6.279e-04,
        4.926e-04, 5.216e-03, 3.892e-03, 3.720e-07, 6.534e-05, 1.983e-03,
        4.390e-04, 4.587e-03, 5.172e-03, 2.977e-04, 3.002e-07, 4.800e-03,
        5.451e-03, 8.110e-03, 7.129e-04, 8.694e-05, 6.572e-04, 7.720e-07,
        2.268e-04, 4.377e-07],
       [6.089e-03, 2.135e-04, 4.548e-04, 2.578e-03, 1.388e-02, 6.110e-04,
        4.177e-04, 4.946e-03, 4.232e-03, 3.724e-06, 3.305e-05, 1.594e-03,
        4.711e-04, 4.710e-03, 4.809e-03, 3.151e-04, 1.272e-06, 4.657e-03,
        6.008e-03, 7.055e-03, 8.514e-04, 8.661e-05, 5.575e-04, 1.420e-06,
        2.936e-04, 3.300e-07]])
'''

# Matrix Product

hv_dist[0, :].reshape(1, -1) @ hv_dist[1, :]
# Output: array([0.066])

np.dot(hv_dist[0, :], hv_dist[1, :])
# Output: 0.06581149298284382

hv_dist[0, :] @ hv_dist[1, :]
# Output: 0.06581149298284382

# Document Cosines

hv_dot = hv_dist @ hv_dist.T
print(hv_dot)

'''
Output:
array([[0.066, 0.066, 0.065, 0.066, 0.065],
       [0.066, 0.066, 0.065, 0.066, 0.065],
       [0.065, 0.065, 0.064, 0.065, 0.064],
       [0.066, 0.066, 0.065, 0.066, 0.065],
       [0.065, 0.065, 0.064, 0.065, 0.065]])
For the vector noms, ||u|| and ||v||, we can use np.linalg.norm(). Here we will break down the computation with elementary operations.
We will apply the Hadamard product to have the square of the coordinates, then sum along the rows, and finally extract the square root:
'''

hv_norm = np.sqrt(np.sum(hv_dist * hv_dist, axis=1))

print(hv_norm)
# Output:
'''
array([0.257, 0.257, 0.253, 0.257, 0.255])
We compute the product of the norms, ||u|| * ||v||, as a matrix product of a column vector by a row vector as with:
'''

hv_norm_pairs = hv_norm.reshape(-1, 1) @ hv_norm.reshape(1, -1)
print(hv_norm_pairs)

'''
Output:
array([[0.066, 0.066, 0.065, 0.066, 0.065],
       [0.066, 0.066, 0.065, 0.066, 0.065],
       [0.065, 0.065, 0.064, 0.065, 0.064],
       [0.066, 0.066, 0.065, 0.066, 0.065],
       [0.065, 0.065, 0.064, 0.065, 0.065]])
We now nearly done with the cosine. We only need to divide the matrix elements by the norm products (u*v)/||u|| * ||v||.
'''

hv_cos = hv_dot / hv_norm_pairs
print(hv_cos) # Output:
'''
array([[1.   , 0.999, 0.997, 0.996, 0.995],
       [0.999, 1.   , 0.997, 0.995, 0.994],
       [0.997, 0.997, 1.   , 0.996, 0.995],
       [0.996, 0.995, 0.996, 1.   , 0.998],
       [0.995, 0.994, 0.995, 0.998, 1.   ]])
'''

# Elementary Mathematical Background for Matrices
'''
A = np.array([[1, 2],
              [3, 4]])
A @ np.array([5, 6])
'''

# Output: array([17, 39])

# Matrices and Rotations
# To finish this notebook, we will have a look at vector rotation. From algebra courses, we know that we can use a matrix to compute a rotation of angle θ. For a two-dimensional vector, the rotation matrix is:

theta_45 = np.pi/4
rot_mat_45 = np.array([[np.cos(theta_45), -np.sin(theta_45)],
                       [np.sin(theta_45), np.cos(theta_45)]])
print(rot_mat_45)

'''
array([[ 0.707, -0.707],
       [ 0.707,  0.707]])
we rotate vector (1, 1) by this angle
'''

rot_mat_45 @ np.array([1, 1])

'''
array([1.110e-16, 1.414e+00])
The matrix of a sequence of rotations, for instance a rotation of 
pi/6 followed by a rotation of pi/4, is simply the matrix product of the individual rotations
R01,R02 = R(01+02), here Rpi/4, Rpi/6 = Rpi/12
'''

theta_30 = np.pi/6
rot_mat_30 = np.array([[np.cos(theta_30), -np.sin(theta_30)],
                       [np.sin(theta_30), np.cos(theta_30)]])
print(rot_mat_30)

'''
array([[ 0.866, -0.5  ],
       [ 0.5  ,  0.866]])
rot_mat_30 @ rot_mat_45
'''

rot_mat_45 @ rot_mat_30
'''
array([[ 0.259, -0.966],
       [ 0.966,  0.259]])
'''

np.arccos(0.25881905)
# Output: 1.3089969339255036

np.pi/4 + np.pi/6
# Output: 1.308996938995747

# Inverting a Matrix

np.linalg.inv(rot_mat_30)

# Output:
'''
array([[ 0.866,  0.5  ],
       [-0.5  ,  0.866]])
'''

np.linalg.inv(rot_mat_30) @ rot_mat_30

'''
Output:
array([[1.000e+00, 7.437e-18],
       [6.295e-17, 1.000e+00]])
'''

torch.linalg.inv(torch.from_numpy(rot_mat_30))

'''
Output:
tensor([[ 0.8660,  0.5000],
        [-0.5000,  0.8660]], dtype=torch.float64)
'''

torch.linalg.inv(torch.from_numpy(rot_mat_30)) @ torch.from_numpy(rot_mat_30)

'''
Output:
tensor([[ 1.0000e+00, -4.0637e-17],
        [ 6.2948e-17,  1.0000e+00]], dtype=torch.float64)
'''

# Application to Neural Networks
# Pytorch

layer1 = torch.nn.Linear(3, 4, bias=False)
print(layer1.weight)

# Output
'''

Parameter containing:
tensor([[-0.4324,  0.0435,  0.1806],
        [-0.5352,  0.0966,  0.2330],
        [-0.2231,  0.5196, -0.0784],
        [-0.2372,  0.1172, -0.3739]], requires_grad=True)
'''
x = torch.tensor([1.0, 2.0, 3.0])
layer1(x)
# Output: tensor([ 0.1963,  0.3571,  0.5810, -1.1245], grad_fn=<SqueezeBackward4>)
layer1.weight @ x
# Output: tensor([ 0.1963,  0.3571,  0.5810, -1.1245], grad_fn=<MvBackward0>)
# Or see: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

x @ layer1.weight.T
# Output: tensor([ 0.1963,  0.3571,  0.5810, -1.1245], grad_fn=<SqueezeBackward4>)



# More Layers
layer1 = torch.nn.Linear(3, 4, bias=False)
layer2 = torch.nn.Linear(4, 2, bias=False)
layer3 = torch.nn.Linear(2, 1, bias=False)

print(layer1.weight, layer2.weight, layer3.weight)

'''
(Parameter containing:
 tensor([[ 0.5711, -0.2106,  0.5642],
         [-0.1257,  0.3728, -0.4489],
         [-0.1961, -0.0592, -0.0630],
         [-0.4868,  0.2738,  0.5165]], requires_grad=True),
 Parameter containing:
 tensor([[ 0.3872,  0.2588,  0.3612, -0.1009],
         [ 0.2428, -0.2557, -0.2185, -0.2425]], requires_grad=True),
 Parameter containing:
 tensor([[-0.7031, -0.4677]], requires_grad=True))
'''
layer3(layer2(layer1(x)))

# Output: tensor([-0.2923], grad_fn=<SqueezeBackward4>)

x @ layer1.weight.T @ layer2.weight.T @ layer3.weight.T
# Output: tensor([-0.2923], grad_fn=<SqueezeBackward4>)





# Automatic Differentiation
 

def f(x, y):
    return x**2 + x * y + y**2
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
z = f(x, y)
print(z)
# Output: tensor(37., grad_fn=<AddBackward0>)
# The retain_graph is necessary to visualize the graph below

z.backward(retain_graph=True)
print(z)
# Output: tensor(37., grad_fn=<AddBackward0>)
print(z.grad_fn)
# Output: <AddBackward0 at 0x12fb25750>
print(z.grad_fn.next_functions)
# Output: ((<AddBackward0 at 0x12fb269e0>, 0), (<PowBackward0 at 0x12fb240d0>, 0))
print(z.grad_fn.next_functions[0][0].next_functions)
# Output: ((<PowBackward0 at 0x12fb24a30>, 0), (<MulBackward0 at 0x12fb26da0>, 0))
print(z.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
#Output: ((<AccumulateGrad at 0x12fb26920>, 0),)
print(z.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)

# Output: ()



x.grad, y.grad
# Output: (tensor(10.), tensor(11.))


from torchviz import make_dot
make_dot(z, params={'x': x, 'y': y, 'z': z}, show_attrs=True).render(
    "autograd_torchviz", format="png")
# Output: 'autograd_torchviz.png'
make_dot(z, params={'x': x, 'y': y, 'z': z}, show_attrs=True)
# Output bilden på hemsidan

# Laddat ner C:\Program Files\Graphviz 
# Lägga till de som miljövariabler i path i systeminställningar