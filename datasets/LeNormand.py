import numpy as np
import torchaudio
import torch
from utils import analyse_file
import os

class LeNormandData(torch.utils.data.Dataset):
    def __init__(self,sr, n_mf,maxlen,classes=[0,1]):
        self.x = []
        self.y = []
        self.sr = sr
        self.n_mfcc = n_mf
        self.maxlen = maxlen
        self.load_data(classes)
    
    def load_data(self,classes):
        for root, dirs, files in os.walk('data/LeNormand/SLI/'):
            for file in files:
                if file.endswith(".wav") and 1 in classes:
                    x,y,powers = analyse_file(os.path.join(root,file),1,1,self.sr,self.maxlen,self.n_mfcc,
                                            timestamp_path=os.path.join(root,file.replace('.wav','.cha')))
                    self.x.append(x)
                    self.y.append(y)
        
        for root, dirs, files in os.walk('data/LeNormand/TD/'):
            for file in files:
                if file.endswith(".wav") and 0 in classes:
                    x,y,powers = analyse_file(os.path.join(root,file),0,1,self.sr,self.maxlen,self.n_mfcc)
                    self.x.append(x)
                    self.y.append(y)

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)
        

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]