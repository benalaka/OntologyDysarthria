import pyAudioAnalysis.audioBasicIO as audioIO
import pyAudioAnalysis.audioSegmentation as aS
import numpy as np
from scipy.io import wavfile

def calculate_energy(segment):
    return sum(np.abs(segment))

def find_strongest_and_weakest_vowels(audio_file_path):
    # Load the audio file
    [Fs, x] = audioIO.read_audio_file(audio_file_path)

    # Segment the audio using a silence-based approach
    segments = aS.silence_removal(x, Fs, 0.020, 0.020, smooth_window=1.0, weight=0.3)

    # Find the segment with the highest and lowest energy
    max_energy = -1
    min_energy = float('inf')
    max_segment = None
    min_segment = None

    for segment in segments:
        energy = calculate_energy(segment)
        if energy > max_energy:
            max_energy = energy
            max_segment = np.array(segment)  # Convert to NumPy array
        if energy < min_energy:
            min_energy = energy
            min_segment = np.array(segment)  # Convert to NumPy array

    return Fs, max_segment, min_segment

# Example usage
audio_file_path = "Log_PhD/wav_dataset/An11DS04F.wav"
Fs, strongest_vowel, weakest_vowel = find_strongest_and_weakest_vowels(audio_file_path)

# You can save these segments as separate audio files if needed using scipy
wavfile.write('strongest_vowel.wav', Fs, strongest_vowel.astype(np.int16))
wavfile.write('weakest_vowel.wav', Fs, weakest_vowel.astype(np.int16))



