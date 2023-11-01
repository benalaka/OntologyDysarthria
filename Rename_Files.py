import os


def rename_wav_files(directory):
    """Renames all WAV files in the specified directory to have the prefix "F".

  Args:
    directory: The directory containing the WAV files to rename.
  """

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            new_filename = "M05_S1_" + filename
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


if __name__ == "__main__":
    directory = "C:/Users/ochie/OneDrive/Documents/PhD/Data/male/M05/Session1/wav_arrayMic"
    rename_wav_files(directory)
