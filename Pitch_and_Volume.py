import scipy.io.wavfile as wavfile
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt
import numpy as np
import librosa

audio2 = "Log_PhD/wav_dataset/An11DS04F.wav"


def calculate_pitch(wav_file):
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(wav_file)

    # Apply Hamming window to the audio data
    hamming_window = hamming(len(audio_data))
    windowed_data = audio_data * hamming_window

    # Perform autocorrelation
    autocorr = np.correlate(windowed_data, windowed_data, mode='full')

    # Find the first peak after the first zero-crossing
    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
    first_zero_crossing = zero_crossings[0]
    peak_index = np.argmax(autocorr[first_zero_crossing:]) + first_zero_crossing

    # Calculate the pitch period in samples
    pitch_period = peak_index - first_zero_crossing

    # Convert pitch period to frequency (Hz)
    pitch_frequency = sample_rate / pitch_period

    return pitch_frequency


def getPitch(file):
    pitch = calculate_pitch(file)
    # print(f"Pitch of {file}: {pitch} Hz")
    return pitch


def getPitchVerdict(file):
    if 0.0 <= getPitch(file) <= 0.25:
        verdict = "Low"
    elif 0.26 <= getPitch(file) <= 0.39:
        verdict = "Normal"
    elif 0.40 <= getPitch(file) <= 1.0:
        verdict = "High"
    else:
        verdict = "Undefined"
    return verdict


def get_volume_in_time(wav_file):
    y, sr = librosa.load(wav_file)
    S = librosa.stft(y, n_fft=2048, hop_length=512)
    mag = np.abs(S)
    power = mag ** 2
    rms = np.sqrt(np.mean(power, axis=1))
    # print(rms)
    return rms


def getAverageVolumeInTime(file):
    v_i_t = get_volume_in_time(file)
    avg_rms = np.mean(v_i_t)
    return avg_rms


def getVolumeInTimeVerdict(file):
    if 0.0 <= getAverageVolumeInTime(file) <= 0.25:
        verdict = "Minimum"
    elif 0.26 <= getAverageVolumeInTime(file) <= 0.39:
        verdict = "Average"
    elif 0.40 <= getAverageVolumeInTime(file) <= 1.0:
        verdict = "Maximum"
    else:
        verdict = "Undefined"
    return verdict


def plotVolumeInTime(file):
    # Get the volume in time of the wav file.
    volume_in_time = get_volume_in_time(file)
    plt.plot(volume_in_time)
    plt.xlabel("Time frames")
    plt.ylabel("RMS amplitude")
    plt.title("Volume in time of audio.wav")
    plt.show()


if __name__ == '__main__':
    print(f"Overall Pitch: {getPitch(audio2)} Hz")
    print("Pitch Verdict",getPitchVerdict(audio2))
    # Get the RMS amplitude of each time frame.
    print("Overall Volume:", getAverageVolumeInTime(audio2), "Hz")
    print("Volume Verdict:", getVolumeInTimeVerdict(audio2))
#     Then plot volume Graph
    plotVolumeInTime(audio2)
