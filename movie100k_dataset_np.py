from torch.utils.data import Dataset
import numpy as np
import pickle

class MovieLensNumpy(Dataset):
    """MovieLensNumpy Numpy Pickle Dataset"""

    def __init__(self, numpy_file, transform=None):
    
        # Load Numpy file
        self.data= np.load(numpy_file)
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        sample = self.data[idx,:]   
        return sample