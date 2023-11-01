import csv
import pandas as pd
import spacy
from spacy.matcher import Matcher
import re
import spotlight
from pandas import *
from numpy import nan
from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD, FOAF
from spotlight import SpotlightException, annotate

nlp = spacy.load('en_core_web_sm')
SERVER = "https://api.dbpedia-spotlight.org/en/annotate"
# Test around with the confidence, and see how many names changes depending on the confidence. However, be aware that anything lower than this (0.83) it will replace James W. McCord and other names that includes James with LeBron James
CONFIDENCE = 0.83

matcher = Matcher(nlp.vocab)
pd.set_option('display.max_colwidth', 200)

CSV_FILE_TRIPLES = 'Log_PhD/dynamic_workspace/all_triples_without_EDA.csv'
ONTOLOGY_FILE = 'Log_PhD/dynamic_workspace/dysarthria_ontology.ttl'

g = Graph()
ex = Namespace("http://dysarthria.org/")
g.bind("ex", ex)

def get_relation(sent):
    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # define the pattern
    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]
    return (span.text)


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod"):
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj"):
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            ## chunk 4
            if tok.dep_.find("obj"):
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]


def get_subject_phrase(doc):
    doc = nlp(doc)
    for token in doc:
        if "subj" in token.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]


def create_triple(sentence):
    entity_pairs = [get_entities(sentence)]

    # extract subject
    subject = get_subject_phrase(sentence)
    # mysubject = [i[0] for i in entity_pairs]
    # subject = mysubject[0]

    # extract relation
    relations = get_relation(sentence)

    # extract object
    mytarget = [i[1] for i in entity_pairs]
    target = mytarget[0]

    return subject, relations, target


def writeTriple(sentence):
    # adding header to csv file
    mytriple = create_triple(sentence)
    headerList = ["the_subject", "relation", "object"]
    with open(CSV_FILE_TRIPLES, 'w') as f:
        writer = csv.writer(f, dialect='excel')
        writerObj = csv.writer(f)
        writerObj.writerow(headerList)
        writerObj.writerow([mytriple[0], mytriple[1], mytriple[2]])
        # s= mytriple[0]
        # v = mytriple[1]
        # o= mytriple[2]
        print(mytriple)
    return mytriple


def annotate_entity(entity, filters={'types': 'DBpedia:Person'}):
    annotations = []
    try:
        annotations = annotate(address=SERVER, text=entity, confidence=CONFIDENCE, filters=filters)
    except SpotlightException as e:
        print(e)
    return annotations


# Function that prepares the values to be added to the graph as a URI or Literal
def prepareValue(row):
    if row == None:  # none type
        value = Literal(row)
    elif isinstance(row, str) and re.match(r'\d{4}-\d{2}-\d{2}', row):  # date
        value = Literal(row, datatype=XSD.date)
    elif isinstance(row, bool):  # boolean value (true / false)
        value = Literal(row, datatype=XSD.boolean)
    elif isinstance(row, int):  # integer
        value = Literal(row, datatype=XSD.integer)
    elif isinstance(row, str):  # string
        value = URIRef(ex + row.replace('"', '').replace(" ", "_").replace(",", "").replace("-", "_"))
    elif isinstance(row, float):  # float
        value = Literal(row, datatype=XSD.float)

    return value


# Convert the non-semantic CSV dataset into a semantic RDF
def csv_to_rdf(df):

    for index, row in df.iterrows():
        id = URIRef(ex + "dysarthria_" + str(index))
        subject = prepareValue(row["the_subject"])
        relation = prepareValue(row["relation"])
        the_object = prepareValue(row["object"])

        # Adds the triples to the graph
        g.add((id, RDF.type, ex.Discourse))
        g.add((id, ex.person_thing, subject))
        g.add((id, ex.relation, relation))
        g.add((id, ex.the_object, the_object))


def writeOntology(csvfile_with_triples):
    # Pandas' read_csv function to load finale_triples.csv
    df = read_csv(csvfile_with_triples)
    # Replaces all instances of nan to None type with numpy's nan
    df = df.replace(nan, None)
    csv_to_rdf(df)
    print(g.serialize())
    g.serialize(destination=ONTOLOGY_FILE, format='turtle')
    return g.serialize()


if __name__ == '__main__':
    writeTriple("It also provides for funds to clear slums and help colleges build dormitories")
    writeOntology(CSV_FILE_TRIPLES)