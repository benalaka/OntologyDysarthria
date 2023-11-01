import os
import shutil

# Define the directories
wav_dataset_dir = "Log_PhD/wav_dataset"
prompt_dataset_dir = "Log_PhD/prompts_dataset"
augmented_prompt_dataset_dir = "Log_PhD/augmented_prompts_dataset"

# Get the list of files in the "wav_dataset" directory
wav_files = os.listdir(wav_dataset_dir)

# Iterate through the files in "wav_dataset" and check for similar names in "augmented_prompt_dataset"
for wav_file in wav_files:
    # Look for a corresponding text file in "augmented_prompt_dataset"
    matching_text_file = os.path.join(augmented_prompt_dataset_dir, os.path.splitext(wav_file)[0] + ".txt")

    # Check if the matching text file exists
    if os.path.exists(matching_text_file):
        # Copy the matching text file to the "prompt_dataset" directory
        shutil.copy(matching_text_file, os.path.join(prompt_dataset_dir, os.path.basename(matching_text_file)))

print("Files copied to 'prompt_dataset'.")
