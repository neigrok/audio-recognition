class Config:
    def __init__(self):
        self.mels_count = 30
        self.frame_count = 30
        self.shape = (self.mels_count, self.frame_count, 1)
        self.sample_rate = 22050
        self.frame_size = 2048
        self.hop = 512
