import numpy as np, pandas as pd
import urllib.request
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import re
from rdflib import Graph, Namespace

# download partial word vectors file
ft_url = "http://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec"
byte_range = 1200000000

req = urllib.request.Request(ft_url)
req.headers['Range'] = f"bytes=0-{byte_range - 1}"

with urllib.request.urlopen(req) as response:
    content = response.read().decode('utf-8')

# parse word vectors into a dictionary
lines = content.strip().split('\n')
print(len(lines))
word_vectors = {}
for line in lines[1:]:
    values = line.split(' ')
    word = values[0]
    vector = np.array(values[1:301], dtype=np.float32) #clip to the 300 columns with embeddings
    word_vectors[word] = vector