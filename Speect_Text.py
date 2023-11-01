# Import Module
import speech_recognition as sr
import os
import datetime

# Get the current timestamp
current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
final_name = current_timestamp + ".txt"
AUDIO_DIRECTORY = "Log_PhD/active_listener/audio" \
                  ""
txtfile = "Log_PhD/active_listener/text/"+ final_name


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
    print("The file with the latest timestamp is:", latest_filename)
    lname = directory + "/" + latest_filename
    return lname

def process_file(file):
    r = sr.Recognizer()
    a = ''
    try:

        with sr.AudioFile(file) as source:
            audio = r.record(source)
            try:
                a = r.recognize_google(audio)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return a
    except:
        print("File error!")


def final_convert(file):
    a = process_file(file)
    with open(txtfile, "w") as file1:
        try:
            file1.write(a)
            print(a)
        except:
            print("Could not convert audio")


final_convert(getLatestFile_Audio(AUDIO_DIRECTORY))
