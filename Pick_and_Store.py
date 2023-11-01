path_prompts = "Log_PhD/augmented_prompts_dataset"
# Log_PhD/augmented_prompts_dataset
source = "Log_PhD/augmented_wav_dataset"
destination = "Log_PhD/wav_dataset"
# augmented_prompts_dataset
# augmented_prompts_dataset
# Import Module
import os
import shutil

# Folder Path
path = path_prompts
path2 = source


# Read text File
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        words = f.read()
        lines = words.split()
        number_of_words = len(lines)
        print(words)
        print(number_of_words)
        return number_of_words


# Change the directory
os.chdir(path)

# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not

    file_path = f"{path}/{file}"
    print(file_path)

    # call read text file function, get file names of prompts with words higher than 3, append a .wav extension
    # and copy to the wav_dataset folder
    try:

        if read_text_file(file_path) >= 3:
            pathname = os.path.basename(file_path)
            full_filename = pathname.split('.')
            filename = full_filename[0]
            print(filename)
            source_path = f"{path2}/{filename}"
            shutil.copy2(source_path, destination)
    except:
        print("File skipped..")
