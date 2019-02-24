import librosa
import numpy as np


class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        
    def preprocess(self, y, sr):
        return y, sr
        
    def get_mels(self, filepath, preprocess=True):
        y, sr = librosa.load(filepath, sr=self.config.sample_rate)
        if preprocess:
            y, sr = self.preprocess(y, sr)
        
        mels = librosa.feature.melspectrogram(
            y,
            n_fft=self.config.frame_size,
            hop_length=self.config.hop,
            n_mels=self.config.mels_count,
            fmax=sr//2
        )
        log_mels = librosa.core.power_to_db(mels, ref=np.max)
        return log_mels
    
    def reshape(self, arr):
        shaped = np.copy(arr)
        N = self.config.shape[1]
        while shaped.shape[1] < N:
            shaped = np.hstack((shaped, shaped))
        # не уверен, но пусть будет
        r_offset = np.random.randint(shaped.shape[1] - N + 1)
        shaped = shaped[:, r_offset: r_offset + N, np.newaxis]
        return shaped