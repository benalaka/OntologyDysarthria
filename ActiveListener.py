import os
import pyaudio
import wave
import time
import datetime

# Get the current timestamp
current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
final_name = current_timestamp + ".wav"
# Set up the audio capture parameters
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds_per_recording = 5  # Adjust as needed

# Set the output directory path
output_directory = "Log_PhD/active_listener/audio"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

p = pyaudio.PyAudio()


def StartTalking():
    try:
        # Create an audio stream
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        print("Listening...")

        frames = []

        # Sleep for 5 seconds while recording
        time.sleep(5)

        # Record audio for 5 seconds
        for _ in range(0, int(fs / chunk * seconds_per_recording)):
            data = stream.read(chunk)
            frames.append(data)

        print("Recording done.")

        # Close the audio stream
        stream.stop_stream()
        stream.close()

        # Save the recorded audio as a WAV file with a timestamp
        file_name = os.path.join(output_directory, final_name)
        wf = wave.open(file_name, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))
        wf.close()

        print(f"Recording saved as {file_name}\n")

    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        p.terminate()
