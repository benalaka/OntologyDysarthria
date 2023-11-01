import os
import pyaudio
import wave
import time
import msvcrt  # For Windows, for detecting key presses

# Set up the audio capture parameters
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
seconds_per_recording = 5  # Adjust as needed

# Set the output directory path
output_directory = os.path.join("Log_PhD", "active_listener")

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

p = pyaudio.PyAudio()

try:
    while True:
        # Create an audio stream
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        print("Listening... (Press 'Y' to stop)")

        frames = []
        start_time = time.time()

        stop_listening = False

        while time.time() - start_time < 5 and not stop_listening:
            data = stream.read(chunk)
            frames.append(data)

            if msvcrt.kbhit():  # Check if a key has been pressed
                key = msvcrt.getch().decode('utf-8').upper()
                if key == 'Y':
                    stop_listening = True
                    print("Stopped listening.")

        # Close the audio stream
        stream.stop_stream()
        stream.close()

        if not stop_listening:
            print("Press 'Enter' to save the recording or 'C' to cancel and exit...")
            user_input = input()

            if user_input == "":
                # Save the recorded audio as a WAV file with a timestamp
                file_name = os.path.join(output_directory, f"recording_{len(os.listdir(output_directory)) + 1}.wav")
                wf = wave.open(file_name, "wb")
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(sample_format))
                wf.setframerate(fs)
                wf.writeframes(b"".join(frames))
                wf.close()

                print(f"Recording saved as {file_name}\n")
            elif user_input.upper() == 'C':
                print("Recording canceled. Exiting.")
                break

except KeyboardInterrupt:
    print("Recording stopped.")
finally:
    p.terminate()
