import pickle
import numpy as np
import re
import pandas as pd
import csv
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from rdflib import Graph, Namespace
import seaborn as sns
import matplotlib.pyplot as plt
from Query_Ontologies import Query_Ontology

# TTL_DESTINATION_PERSON_MISSING = 'Log_PhD/ontology/no_person_thing_short.ttl'
SEMANTIC_PROJECTION_ONTOLOGY = 'Log_PhD/dynamic_workspace/semantic_projection_ontology.ttl'
TTL_DESTINATION_PREDICATE_MISSING = 'Log_PhD/ontology/no_predicate.ttl'
TTL_DESTINATION_OBJECT_MISSING = 'Log_PhD/ontology/no_object.ttl'
MY_TXT_IN = 'Log_PhD/ontology/staged_sentences.txt'
MY_SEMANTIC_TXTS_OUT = 'Log_PhD/ontology/staged_sentences.txt'
FINAL_TXT_OUT = '/content/drive/MyDrive/Log_PhD/Ontology/final_staged_sentences.txt'
INTENT_SOURCE_CSV = '/content/drive/MyDrive/Log_PhD/Ontology/bert_intent.csv'
ONTOLOGY_FILE = 'Log_PhD/dynamic_workspace/dysarthria_ontology.ttl'
LARGE_ONTOLOGY_FILE = 'Log_PhD/ontology/no_person_thing.ttl'
CSV_OUT_FOR_LINK_PREDICTION = 'Log_PhD/dynamic_workspace/link_pred_candidates.csv'

# Name of the serialized word vectors file
cached_vectors_file = "Log_PhD/semantic_projection/word_vectors.pkl"

# Load word vectors from the pickle file
with open(cached_vectors_file, 'rb') as f:
    word_vectors = pickle.load(f)


def get_missing_subject_anchor(missing_person_file):
    g = Graph()
    g.parse(missing_person_file, format="ttl")

    # Define the namespace
    ns1 = Namespace("http://example.org/")
    # Query the namespace values
    query = f"SELECT ?value WHERE {{ ?x ns1:person_thing ?value }}"
    results = g.query(query, initNs={"ns1": ns1})

    anchor_1 = None  # Default value
    anchor_2 = None  # Default value
    size = None  # Default

    # define the anchors based on placeholders
    if "happy" in [str(result[0]) for result in results] or "sad" in [str(result[0]) for result in
                                                                      results] or "angry" in [str(result[0]) for result
                                                                                              in results]:
        anchor_1 = word_vectors['happy']
        anchor_2 = word_vectors['angry']
        # anchor words for addition and subtraction
        size = {
            'add': ['happy', 'joyful', 'excited'],
            'subtract': ['sad', 'gloomy', 'angry']
        }
    elif "statement" in [str(result[0]) for result in results] or "emphasis" in [str(result[0]) for result in
                                                                                 results] or "whquestion" in [
        str(result[0]) for result in results] or "ynquestion" in [str(result[0]) for result in results]:
        anchor_1 = word_vectors['statement']
        anchor_2 = word_vectors['question']
        # anchor words for addition and subtraction
        size = {
            'add': ['statement', 'neutral', 'emphasis'],
            'subtract': ['clarification', 'concern', 'question']
        }
    return anchor_1, anchor_2, size


def get_missing_object_anchor(missing_person_file):
    g = Graph()
    g.parse(missing_person_file, format="ttl")

    # Define the namespace
    ns1 = Namespace("http://example.org/")
    # Query the namespace values
    query = f"SELECT ?value WHERE {{ ?x ns1:the_object ?value }}"
    results = g.query(query, initNs={"ns1": ns1})

    anchor_1 = None  # Default value
    anchor_2 = None  # Default value
    size = None  # Default

    # define the anchors based on placeholders
    if "happy" in [str(result[0]) for result in results] or "sad" in [str(result[0]) for result in
                                                                      results] or "angry" in [str(result[0]) for result
                                                                                              in results]:
        anchor_1 = word_vectors['happy']
        anchor_2 = word_vectors['angry']
        # anchor words for addition and subtraction
        size = {
            'add': ['happy', 'joyful', 'excited'],
            'subtract': ['sad', 'gloomy', 'angry']
        }
    elif "statement" in [str(result[0]) for result in results] or "emphasis" in [str(result[0]) for result in
                                                                                 results] or "whquestion" in [
        str(result[0]) for result in results] or "ynquestion" in [str(result[0]) for result in results]:
        anchor_1 = word_vectors['statement']
        anchor_2 = word_vectors['question']
        # anchor words for addition and subtraction
        size = {
            'add': ['statement', 'neutral', 'emphasis'],
            'subtract': ['clarification', 'concern', 'question']
        }
    return anchor_1, anchor_2, size


a1 = get_missing_subject_anchor(SEMANTIC_PROJECTION_ONTOLOGY)[0]
a2 = get_missing_subject_anchor(SEMANTIC_PROJECTION_ONTOLOGY)[1]
s = get_missing_subject_anchor(SEMANTIC_PROJECTION_ONTOLOGY)[2]


def is_url(value):
    if value.startswith("http://") or value.startswith("https://"):
        return True
    else:
        return False


def get_words_missing_subject(missing_person_file):
    # Load the TTL file into an RDF graph
    g = Graph()
    g.parse(missing_person_file, format="ttl")

    # Define the namespace
    ns1 = Namespace("http://example.org/")

    # Query the namespace values
    query = """
      SELECT ?relation ?object
      WHERE {
          ?x ns1:relation ?relation ;
            ns1:the_object ?object .
      }
  """
    results = g.query(query, initNs={"ns1": ns1})

    # Extract the values, replace underscores, and store them in the NumPy array
    words = []
    final_words = []
    for result in results:
        # Process relation value
        if "http" in (result[0]):
            relation = relation.split("/")[-1].strip()
        else:
            relation = str(result[0]).replace("_", " ")
            relation = re.sub('  ', ' ', relation)
        the_object = str(result[1]).replace("_", " ")
        words.extend([relation, the_object])

    # Convert the list to a NumPy array
    words = np.array(words)

    for myword in words:
        # print(myword.split("/")[-1].strip())
        final_words.extend([myword.split("/")[-1].strip()])
    final_words = np.array(final_words)
    # Print the array
    return final_words


def get_clean_missing_person(missing_person_file):
    missing_person_words = get_words_missing_subject(missing_person_file)
    # print(missing_person_words)

    atomic_words = []

    for x in missing_person_words:
        x = re.split(r"\s+", x)
        atomic_words.extend(x)

    atomic_words = np.array(atomic_words)

    atomic_words = np.unique(atomic_words)

    # Remove empty string elements
    atomic_words = atomic_words[atomic_words != '']

    # Remove numbers from the array
    atomic_words = np.array([item for item in atomic_words if not re.search(r'\d', item)])

    # Convert strings to lowercase to avoid keyerror
    atomic_words = np.char.lower(atomic_words)

    return atomic_words


# calculates the semantic direction
def get_direction(anchors, wv):
    add_vectors = np.vstack([wv[word] for word in anchors['add']])
    subtract_vectors = np.vstack([wv[word] for word in anchors['subtract']])

    add_direction = np.mean(add_vectors, axis=0)
    subtract_direction = np.mean(subtract_vectors, axis=0)

    sem_dir = add_direction - subtract_direction
    return sem_dir


sem_dir = get_direction(anchors=s, wv=word_vectors)
sem_dir.shape

# Calculate semantic direction
sem_dir = a1 - a2


def get_df_sim(missing_person_file):
    # words = ["pipe", "home", "cat", "school", "grandfather", "court", "church", "club"]
    words = get_clean_missing_person(missing_person_file)
    words_vecs = np.array([word_vectors[w] for w in words])
    words_vecs_shape = words_vecs.shape
    sims = cosine_similarity([sem_dir], words_vecs)
    df_sims = pd.DataFrame(dict(direction=sims[0], term=words))

    return df_sims


def get_df_sim_fresh(fresh):
    words_vecs = np.array([word_vectors[w] for w in fresh])
    words_vecs_shape = words_vecs.shape
    sims = cosine_similarity([sem_dir], words_vecs)
    df_sims = pd.DataFrame(dict(direction=sims[0], term=fresh))

    return df_sims


def plot_Semantic_Direction(missing_ontology):
    # Reorder the terms based on direction
    df_sims = get_df_sim(missing_ontology)
    # print(df_sims)
    df_sims_sorted = df_sims.sort_values('direction')

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    # sns.barplot(data=df_sims_sorted, x='term', y='direction', palette='viridis')
    sns.barplot(data=df_sims_sorted, x='term', y='direction', hue='direction', palette='viridis', legend=False)
    plt.xticks(rotation=90)
    plt.xlabel('Term')
    plt.ylabel('Direction')
    plt.title('Semantic Direction')
    return plt.show()


def order_terms_per_direction(ontology_filein):
    # Reorder the terms based on direction
    df_sims = get_df_sim(ontology_filein)
    # print(df_sims)
    df_sims_sorted = df_sims.sort_values('direction')
    return df_sims_sorted


def write_to_Destination(file):
    myfile = open(file, 'w')
    # Open the CSV File that is the primary file for the intent prediction (BERT) class
    with open(INTENT_SOURCE_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["text", "intent"])


def get_Universal_Data_Frame():
    df_bigger = pd.DataFrame(order_terms_per_direction(LARGE_ONTOLOGY_FILE))
    return df_bigger


def get_smaller_Data_Frame():
    df_smaller = pd.DataFrame(get_df_simsi_sorted())
    return df_smaller


def get_triple(staged_file_in):
    with open(staged_file_in, 'r') as file:
        for line in file:
            fullsentence = line.split(" ", 1)[1]
            words = fullsentence.split()
            fresh_triple = [word.lower() for word in words]
            return fresh_triple


def EDA(staged_file_in):
    with open(staged_file_in, 'r') as file:
        for line in file:
            EDA = line.split()[0]
    return EDA


def get_df_simsi_sorted():
    df_simsi = get_df_sim_fresh(get_triple(MY_TXT_IN))
    # print(df_simsi)
    df_simsi_sorted = df_simsi.sort_values('direction')
    return df_simsi_sorted


def generate_semantic_projection():
    closeness = 0.01
    r = Query_Ontology(SEMANTIC_PROJECTION_ONTOLOGY)[0]
    o = Query_Ontology(SEMANTIC_PROJECTION_ONTOLOGY)[1]
    # Loop through the smaller dataframe
    matched_terms = []
    for idx_s, row_s in get_smaller_Data_Frame().iterrows():
        term_s = row_s['term']
        direction_s = row_s['direction']

        # Loop through the bigger dataframe
        for idx_b, row_b in get_Universal_Data_Frame().iterrows():
            term_b = row_b['term']
            direction_b = row_b['direction']

            # Check the closeness of 'direction' values
            if abs(direction_s - direction_b) <= closeness:
                # matched_terms.append(term_b)
                matched_terms.append(term_b + " " + str(abs(direction_s - direction_b)))

    # print("Matched terms:", matched_terms)

    # Separate the terms and similarities
    terms, similarities = zip(*[term_sim.split() for term_sim in matched_terms])

    # Create the DataFrame
    df = pd.DataFrame({'term': terms, 'similarity': similarities})

    # Convert the 'similarity' column to numeric type (float)
    df['similarity'] = df['similarity'].astype(float)

    # Sort the DataFrame by 'similarity' from smallest to largest
    df_sorted = df.sort_values(by='similarity')

    # Reset the index to have consecutive integer indices
    df_sorted = df_sorted.reset_index(drop=True)

    # final_df = df_sorted.head(7)
    final_df1 = df_sorted.head(8)
    final_df = final_df1.tail(4)
    print("SEMANTIC PROJECTION COMPLETE...BELOW ARE CANDIDATE WORDS (alongside similarity scores) FROM SEMANTIC PROJECTION")
    print(final_df)
    print("BELOW ARE CANDIDATE TRIPLES FOR ONTOLOGY LEARNING (LINK PREDICTION)")
    # Start writing to csv
    headerList = ["s", "p", "o"]
    with open(CSV_OUT_FOR_LINK_PREDICTION, 'w', newline='') as f:
        writerObj = csv.writer(f)
        writerObj.writerow(headerList)

    for item in final_df['term']:
        print(item,r,o)
        with open(CSV_OUT_FOR_LINK_PREDICTION, 'a', newline='') as f:
            writerObj = csv.writer(f)
            writerObj.writerow([item, r, o])




# check if set of triples returned are nouns (FOR LATER USE)
def is_noun(word):
    # Part of speech tagging using NLTK
    pos_tags = nltk.pos_tag([word])
    pos_tag = pos_tags[0][1]

    # Check if the POS tag is a noun
    return pos_tag.startswith('NN')


if __name__ == '__main__':
    # plot_Semantic_Direction(TTL_DESTINATION_PERSON_MISSING)
    # print(order_terms_per_direction(SEMANTIC_PROJECTION_ONTOLOGY))
    generate_semantic_projection()
