from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import albumentations
from augment import MelSpectrogram, SpecAugment, SpectToImage
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, PolarityInversion, Gain, AddGaussianSNR
import os
import librosa
import numpy as np


data_path = 'D:\\Desktop\\competition\\RFCX\\data'
num_class = 24
fft = 2048
hop = 512
sr = 48000
length = 10 * sr
melspectrogram_parameters = {
        "n_mels": 256,
        'n_fft': 2048, 
        'hop_length': 512,
        'fmin': 84, 
        'fmax': 15056 
    }

sound_augment = Compose([
    PolarityInversion(p=0.2),
    Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.3),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
    AddGaussianSNR(p=0.2)
#     TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2)
#     Shift(min_fraction=-0.1, max_fraction=0.1, p=0.2),
])

spec_augment = albumentations.Compose([
    MelSpectrogram(parameters=melspectrogram_parameters, always_apply=True),
    SpecAugment(p=0.2),
    SpectToImage(always_apply=True)
])

to_image = albumentations.Compose([
    MelSpectrogram(parameters=melspectrogram_parameters, always_apply=True),
    SpectToImage(always_apply=True)
])

def augment(wav):
    data = sound_augment(samples=wav, sample_rate=sr), sr
    image = spec_augment(data=data)['data']
    return image.transpose(2, 1, 0)

def get_image(wav):
    data = wav, sr
    image = to_image(data=data)['data']
    return image.transpose(2, 1, 0)



class TrainDataset(Dataset):
    def __init__(self, data_df, is_valid=False):
        self.data_df = data_df
        self.is_valid = is_valid
        self.transformer = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        s = self.data_df.iloc[idx]
        img = get_image(s['data'])
#         if self.is_valid:
#             img = get_image(s['data'])
#         else:
#             img = augment(s['data'])
        return torch.tensor(img, dtype=torch.float32), s['species_id']


class TestDataset(Dataset):
    def __init__(self, test_files):
        self.test_files = test_files 
        self.transformer = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.test_files)
    
    def __getitem__(self, idx):
        audio_file_path = os.path.join(data_path, 'test', self.test_files[idx])
        wav, _ = librosa.load(audio_file_path, sr=sr)
        segments = len(wav) / length
        segments = int(np.ceil(segments))
        img = []
        for i in range(0, segments):
            # Last segment going from the end
            if (i + 1) * length > len(wav):
                slice = wav[len(wav) - length:len(wav)]
            else:
                slice = wav[i * length:(i + 1) * length]
            img.append(get_image(slice))
        return torch.tensor(img, dtype=torch.float32)


