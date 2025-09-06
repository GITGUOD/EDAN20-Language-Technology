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