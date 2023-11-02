import os


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
    # print("The file with the latest timestamp is:", lname)
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
    # print("The file with the latest timestamp is:", latest_filename)
    lname = directory + "/" + latest_filename
    return lname
