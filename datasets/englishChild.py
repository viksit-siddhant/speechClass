import numpy as np
import torchaudio
import torch
from speechClass.utils import analyse_file
import os

class EnglishData(torch.utils.data.Dataset):
    def __init__(self,sr, n_mf,maxlen,path='data/english_children'):
        self.x = []
        self.y = []
        self.sr = sr
        self.n_mfcc = n_mf
        self.maxlen = maxlen
        self.load_data(path)

    def load_data(self,path):
        for root,dirs,files in os.walk(path):
            for file in files:
                if file.endswith('.wav'):
                    x,y,powers = analyse_file(os.path.join(root,file),0,1,self.sr,self.maxlen,self.n_mfcc)
                    self.x.append(x)
                    self.y.append(y)
        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]