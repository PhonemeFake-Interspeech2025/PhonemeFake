import numpy as np
import librosa
import os
import soundfile as sf


def peak_normalization(data_path):
    audio_files = os.listdir(data_path)
    for file in audio_files:
        file_path = os.path.join(data_path, file)
        
        y, sr = librosa.load(file_path, sr=16000)
        y_norm = y / np.max(np.abs(y))
        sf.write(file_path, y_norm, sr)


if __name__ == "__main__":
    peak_normalization("/home/ugrad-su24/ege/PhonemeFake/gen_ITW_minibatch_200")