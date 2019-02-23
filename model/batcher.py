from keras.utils import Sequence


class Batcher(Sequence):
    def __init__(self, x, y, batch_size=40):
        self.batch_size = batch_size
        self.x = np.array([prepare_shape(f['log_mel'], config)  for f in data])
        self.y = np.array([f['type'] for f in data])
        self.indexes = np.arange(len(data))
        
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, index):
        indexes_tmp = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        res_x = self.x[indexes_tmp]
        res_y = self.y[indexes_tmp]
        return res_x, res_y
