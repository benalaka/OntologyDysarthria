from ActiveListener import StartTalking
from Speect_Text import final_convert
from DysarthriaEmotion import getEmotion
from DialogueActs import get_DialogueAct_Text
from Generate_Ontology import writeTriple, writeOntology
from Latest_File import getLatestFile_Text,getLatestFile_Audio
from Pitch_and_Volume import plotVolumeInTime,get_volume_in_time,getPitch,getAverageVolumeInTime

DIRNAME_AUDIO = "Log_PhD/active_listener/audio"
DIRNAME_TEXT = "Log_PhD/active_listener/text"
CSV_FILE_TRIPLES = 'Log_PhD/dynamic_workspace/all_triples_without_EDA.csv'


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
# FROM THIS POINT ON, WE CONTROL THE SCRIPTS FROM OUTSIDE
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
# 6. Ontology with EDA
writeOntology(CSV_FILE_TRIPLES)
print("_________________________________________________________________________________________________________________\n")

print("###############################################\n SHOW PITCH, VOLUME AND PLOTTING AUDIO VOLUME IN TIME\n###############################################\n")
# 7. Get Volume Plot
getPitch(getLatestFile_Audio(DIRNAME_AUDIO))
getAverageVolumeInTime(getLatestFile_Audio(DIRNAME_AUDIO))
plotVolumeInTime(getLatestFile_Audio(DIRNAME_AUDIO))

print("_________________________________________________________________________________________________________________\n")