import csv
import pandas as pd

TTL_DESTINATION_PERSON_MISSING = 'Log_PhD/ontology/no_person_thing_short.ttl'
TTL_DESTINATION_PREDICATE_MISSING = 'Log_PhD/ontology/no_predicate.ttl'
TTL_DESTINATION_OBJECT_MISSING = 'Log_PhD/ontology/no_object.ttl'
MY_TXT_IN = 'Log_PhD/ontology/staged_sentences.txt'
MY_SEMANTIC_TXTS_OUT = 'Log_PhD/ontology/INTENT_staged_sentences.txt'
FINAL_TXT_OUT = 'Log_PhD/ontology/final_staged_sentences.txt'
INTENT_SOURCE_CSV = '/content/drive/MyDrive/Log_PhD/Ontology/bert_intent.csv'



def getTriple(filein):
    # Reorder the terms based on direction
    with open(filein, 'r') as file:
        for line in file:
            # print(str(line.strip()))  # Convert line to a string and remove leading/trailing whitespaces
            fullsentence = line.strip()
            # Step 1: Split the sentence into words using the split() method.
            words = fullsentence.split()
            # Step 2: Convert each word to lowercase as per the desired output.
            fresh_triple = [word.lower() for word in words]
    return fresh_triple


def getSentence(filein):
    with open(filein, 'r') as file:
        for line in file:
            fullsentence = line.strip()
            sentence = str(fullsentence.strip())
    return sentence


def getEDA(filein):
    with open(filein, 'r') as file:
        for line in file:
            # get the EDA
            EDA_word = line.split()

            if EDA_word:
                EDA = EDA_word[0]
            else:
                print("No words found in the line.")
    return EDA

if __name__ == '__main__':
    print(getSentence(MY_TXT_IN))