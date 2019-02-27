import librosa
import numpy as np


class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        
    def preprocess(self, y, sr):
        times = librosa.effects.split(y, top_db=30)
        ys = [y[start:end] for start, end in times]
        ys = filter(lambda x: len(x) > self.config.frame_size, ys)
        return ys, sr
        
    def get_mels(self, filepath):
        orig_y, sr = librosa.load(filepath, sr=self.config.sample_rate)
        ys, sr = self.preprocess(orig_y, sr)
        for y in ys:
            mels = librosa.feature.melspectrogram(
                y,
                n_fft=self.config.frame_size,
                hop_length=self.config.hop,
                n_mels=self.config.mels_count,
                fmax=sr//2
            )
            log_mels = librosa.core.power_to_db(mels, ref=np.max)
            yield log_mels
    
    def reshape(self, arr):
        shaped = np.copy(arr)
        N = self.config.shape[1]
        while shaped.shape[1] < N:
            shaped = np.hstack((shaped, shaped))
        # не уверен, но пусть будет
        r_offset = np.random.randint(shaped.shape[1] - N + 1)
        shaped = shaped[:, r_offset: r_offset + N, np.newaxis]
        return shaped