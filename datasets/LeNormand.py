import numpy as np
from speechClass.utils import analyse_file
import torch
import os

class LeNormandData(torch.utils.data.Dataset):
    def __init__(self,sr, n_mf,maxlen,classes=[0,1],path='data/LeNormand/'):
        self.x = []
        self.y = []
        self.sr = sr
        self.n_mfcc = n_mf
        self.maxlen = maxlen
        self.load_data(classes,path)
    
    def load_data(self,classes,path):
        for root, dirs, files in os.walk(os.path.join(path,'SLI/')):
            for file in files:
                if file.endswith(".wav") and 1 in classes:
                    x,y,powers = analyse_file(os.path.join(root,file),1,1,self.sr,self.maxlen,self.n_mfcc,
                                            timestamp_path=os.path.join(root,file.replace('.wav','.cha')))
                    self.x.append(x)
                    self.y.append(y)
        
        for root, dirs, files in os.walk(os.path.join(path,'TD/')):
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