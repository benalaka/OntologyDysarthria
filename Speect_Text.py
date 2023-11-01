path_audio = "Log_PhD/wav"
new_path_audio = "Log_PhD/augmented_wav_dataset"
prompts = "Log_PhD/augmented_prompts_dataset"
myoutput = "Log_PhD/data_subject/recognized.csv"

# Import Module
import speech_recognition as sr
import os

# Folder Path
path_audio = new_path_audio
path_prompts = prompts

DIRNAME = new_path_audio
OUTPUTFILE = myoutput


def get_file_paths(dirname):
    file_paths = []
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


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


def main():
    files = get_file_paths(DIRNAME)  # get all file-paths of all files in dirname and subdirectories
    for file in files:  # execute for each file
        (filepath, ext) = os.path.splitext(file)  # get the file extension
        file_name = os.path.basename(file)  # get the basename for writing to output file
        full_filename = file_name.split('.')
        txtfile = full_filename[0] + ".txt"
        if ext == '.wav':  # only interested if extension is '.wav'
            a = process_file(file)  # result is returned to a

            with open(os.path.join(prompts, txtfile), "w") as file1:
                try:
                    file1.write(a)
                except:
                    print("File error!")


if __name__ == '__main__':
    main()
