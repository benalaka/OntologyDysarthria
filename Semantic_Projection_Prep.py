import urllib.request
import pickle
import os
import numpy as np

# Constants
ft_url = "http://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec"
byte_range = 1200000000
cached_file = "wiki.en.vec"  # Name of the locally cached file
cached_vectors_file = "Log_PhD/semantic_projection/word_vectors.pkl"  # Name of the serialized word vectors file

# Check if the cached word vectors file exists
if os.path.exists(cached_vectors_file):
    # Load word vectors from the cached file
    with open(cached_vectors_file, 'rb') as f:
        word_vectors = pickle.load(f)
else:
    # Download and save the .vec file
    if not os.path.exists(cached_file):
        req = urllib.request.Request(ft_url)
        req.headers['Range'] = f"bytes=0-{byte_range - 1}"
        with urllib.request.urlopen(req) as response:
            content = response.read()
        with open(cached_file, 'wb') as f:
            f.write(content)

    # Parse word vectors into a dictionary
    with open(cached_file, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    print(len(lines))
    word_vectors = {}
    for line in lines[1:]:
        values = line.split(' ')
        word = values[0]
        vector = np.array(values[1:301], dtype=np.float32)  # Clip to the 300 columns with embeddings
        word_vectors[word] = vector

    # Serialize and save the word vectors to a file
    with open(cached_vectors_file, 'wb') as f:
        pickle.dump(word_vectors, f)

# Now you can use the 'word_vectors' dictionary as needed
