import librosa
import soundfile
import numpy as np
from pydub import AudioSegment
import pickle

path_audio = "Log_PhD/wav_dataset"
CSV_FILE = 'Log_PhD/data_emotion/emotion.csv'
MODEL = 'Log_PhD/data_emotion/mlp_classifier.model'
audio2 = "Log_PhD/wav_dataset/An11DS04F.wav"

MYMODEL = pickle.load(open(MODEL, "rb"))

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            # noinspection PyUnresolvedReferences
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            # noinspection PyUnresolvedReferences
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            # noinspection PyUnresolvedReferences
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            # noinspection PyUnresolvedReferences
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            # noinspection PyUnresolvedReferences
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result


def getEmotion(file):
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(1)
    sound.export(file, format="wav")
    features = extract_feature(file, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = MYMODEL.predict(features)[0]
    # show the result !
    print(file, result)
    return result


if __name__ == '__main__':
    getEmotion(audio2)
