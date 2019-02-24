class Config:
    def __init__(self):
        self.mels_count = 60
        self.frame_count = 200
        self.shape = (self.mels_count, self.frame_count, 1)
        self.sample_rate = 22050
        self.frame_size = 1024
        self.hop = self.frame_size // 2
