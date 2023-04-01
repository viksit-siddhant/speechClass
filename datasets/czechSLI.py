from speechClass.utils import analyse_file
import numpy as np
import torch
import os

class czechData(torch.utils.data.Dataset):
    def __init__(self,sr,n_mels,maxlen,path='data/czechSLI/'):
        self.sr = sr
        self.n_mels = n_mels
        self.maxlen = maxlen
        self.x = []
        self.y = []
        self.num_pos = 0
        self.load_data(path)
    
    def load_data(self,path):
        for root,dirs,files in os.walk(os.path.join(path,'Healthy')):
            for file in files:
                if file.endswith('.wav'):
                    x,y,powers = analyse_file(os.path.join(root,file),0,1,target_sr=self.sr,maxlen=self.maxlen,n_fft=self.n_mels)
                    self.x.append(x)
                    self.num_pos += x.shape[0]
                    self.y.append(y)
        
        for root,dirs,files in os.walk(os.path.join(path,'Patients')):
            for file in files:
                if file.endswith('.wav'):
                    x,y,powers = analyse_file(os.path.join(root,file),1,1,target_sr=self.sr,maxlen=self.maxlen,n_fft=self.n_mels)
                    self.x.append(x)
                    self.y.append(y)

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]