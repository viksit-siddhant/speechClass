import numpy as np
import torchaudio
import torch
import os

def analyse_file(audio_path,
                 target_class,
                 len_labels,
                 target_sr=32000,
                 maxlen  = None, 
                 n_fft = 512,
                 threshold = 0,
                 timestamp_path = None,
                 ):
    '''
    Returns a tuple of (xes,yes,powers) where:
    xes is a numpy array of shape (num_samples,1,n_mfcc,maxlen)
    yes is a numpy array of shape (num_samples,len_labels)
    powers is a numpy array of shape (num_samples,)
    '''
    if maxlen is None:
        maxlen = n_fft//2 + 1
    audio,sr  = torchaudio.load(audio_path)
    if audio is None or audio.shape[-1] < 256:
        return np.zeros((0,1,n_fft//2+1,maxlen)),np.zeros((0,len_labels)),np.zeros((0,))
    audio = torchaudio.transforms.Resample(sr,target_sr)(audio)
    
    if audio.shape[0] > 1:
        audio = torch.mean(audio,dim=0).unsqueeze(0)

    if timestamp_path is not None:
        audio_segments = [] 
        timestamps = get_timestamps(timestamp_path)
        for start,end in timestamps:
            start = int(start*target_sr/1000)
            end = int(end*target_sr/1000)
            if end < audio.shape[-1]:
                audio_segments.append(audio[:,start:end])
        audio = torch.cat(audio_segments,dim=1)

    xes = []
    yes = []
    powers = []
    mfcc = torchaudio.transforms.Spectrogram(n_fft=n_fft)(audio).numpy()
    num_samples = int(np.ceil(mfcc.shape[-1]/maxlen))
    for i in range(num_samples):
        sample = np.zeros((1,n_fft//2+1,maxlen))
        img = mfcc[:,:,i*maxlen : (i+1)*maxlen]
        sample[:,:,:img.shape[-1]] = img
        powers.append(np.sum(np.square(sample)))
        if np.sum(np.square(sample)) >= threshold:
            xes.append(sample)

    xes = np.concatenate(xes)
    if xes.shape[0] == 0:
        xes = np.zeros((0,1,n_fft//2+1,maxlen))
        yes = np.zeros((0,len_labels))
        powers = np.zeros((0,))
        return xes,yes,powers
    xes = np.expand_dims(xes,axis=1)
    
    yes = np.zeros((num_samples,len_labels))
    if len_labels == 1:
        yes[:,0] = target_class
    else:
        yes[:,target_class] = 1
    return xes,yes,powers

def get_timestamps(path, token = "*CHI"):
    '''
    Returns a list of timestamps from CHAT file with each element being (start,end) in miliseconds
    '''
    timestamps = []
    with open(path, 'r') as f:
        for line in f:
            try:
                if line.startswith(token):
                    timestamps.append([int(x) for x in line.split()[-1][1:-1].split('_')])
            except:
                pass
    return timestamps

def serialize_data(path):
    from speechClass.datasets.englishChild import EnglishData
    from speechClass.datasets.czechSLI import CzechData
    from speechClass.datasets.LeNormand import LeNormandData

    english_data = EnglishData(32000,512,None,path='speechClass/data/english_children')
    czech_data = CzechData(32000,512,None,path='speechClass/data/czechSLI')
    lenormand_data = LeNormandData(32000,512,None,path='speechClass/data/LeNormand')
    f = open(path, 'wb')
    np.savez(f,
             english_x = english_data.x,
             english_y = english_data.y,
             czech_x = czech_data.x,
             czech_y = czech_data.y,
             lenormand_x = lenormand_data.x,
             lenormand_y = lenormand_data.y,)


class Dataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
