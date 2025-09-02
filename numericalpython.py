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