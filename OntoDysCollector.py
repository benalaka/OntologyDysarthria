from ActiveListener import StartTalking
from Speect_Text import final_convert
from DysarthriaEmotion import getEmotion
from DialogueActs import get_DialogueAct_Text
from Generate_Ontology import writeTriple, writeOntology
import os

DIRNAME_AUDIO = "Log_PhD/active_listener/audio"
DIRNAME_TEXT = "Log_PhD/active_listener/text"
CSV_FILE_TRIPLES = 'Log_PhD/dynamic_workspace/all_triples_without_EDA.csv'


def getLatestFile_Audio(directory):
    # Get a list of all files in the directory
    file_list = os.listdir(directory)

    # Initialize variables to keep track of the latest timestamp and filename
    latest_timestamp = None
    latest_filename = None

    # Iterate through the files in the directory
    for filename in file_list:
        if filename.endswith('.wav'):
            # Extract the timestamp from the filename
            timestamp_str = filename.split('.')[0]  # Remove the ".wav" extension
            timestamp = timestamp_str.replace('_', ' ')

            # Compare the timestamp with the latest one found so far
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_filename = filename

    # Print the latest filename
    lname = directory + "/" + latest_filename
    print("The file with the latest timestamp is:", lname)
    return lname


def getLatestFile_Text(directory):
    # Get a list of all files in the directory
    file_list = os.listdir(directory)

    # Initialize variables to keep track of the latest timestamp and filename
    latest_timestamp = None
    latest_filename = None

    # Iterate through the files in the directory
    for filename in file_list:
        if filename.endswith('.txt'):
            # Extract the timestamp from the filename
            timestamp_str = filename.split('.')[0]  # Remove the ".txt" extension
            timestamp = timestamp_str.replace('_', ' ')

            # Compare the timestamp with the latest one found so far
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_filename = filename

    # Print the latest filename
    print("The file with the latest timestamp is:", latest_filename)
    lname = directory + "/" + latest_filename
    return lname


def readText(file_path):
    try:
        with open(file_path, 'r') as file:
            file_contents = file.read()
            # Process the file contents here (e.g., print or manipulate the data)
            print(file_contents)
    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
    except IOError as e:
        print(f"An error occurred while reading the file: {str(e)}")
    return file_contents


###############################
# FROM THIS POINT WE CONTROL THE SCRIPTS FROM OUTSIDE
############################

print("###############################################\n START TALKING\n###############################################\n")

# 1. Listener to Patient
# StartTalking()
print("_________________________________________________________________________________________________________________\n")

# 2. Convert their Speech
print("###############################################\n CONVERTING SPEECH TO TEXT\n###############################################\n")
final_convert(getLatestFile_Audio(DIRNAME_AUDIO))
print("_________________________________________________________________________________________________________________\n")

print("###############################################\n FETCHING SPEECH EMOTION\n###############################################\n")
# 3. Get Emotion
getEmotion(getLatestFile_Audio(DIRNAME_AUDIO))
print("_________________________________________________________________________________________________________________\n")

print("###############################################\n RETRIEVING DIALOGUE ACTS\n###############################################\n")
# 4. Get Dialogue Act
get_DialogueAct_Text(getLatestFile_Text(DIRNAME_TEXT))
print("_________________________________________________________________________________________________________________\n")

print("###############################################\n EXTRACTING SLIM TRIPLES (Without EDA)\n###############################################\n")
# 5. Get Triples without the EDA
writeTriple(readText(getLatestFile_Text(DIRNAME_TEXT)))
print("_________________________________________________________________________________________________________________\n")

print("###############################################\n BUIDLING SLIM ONTOLOGY (Without EDA)\n###############################################\n")
# 6. Ontology without EDA
writeOntology(CSV_FILE_TRIPLES)
print("_________________________________________________________________________________________________________________\n")

# 7. Ontology with EDA
