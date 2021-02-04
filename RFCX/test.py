import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import KFold,StratifiedKFold
from skimage.transform import resize
import os
import gc
import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataset import TrainDataset, TestDataset
from model import ResNest


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
GLOBAL_SEED = 42
setup_seed(GLOBAL_SEED)

data_path = 'D:\\Desktop\\competition\\RFCX\\data'
feat_path = 'D:\\Desktop\\competition\\RFCX\\features'
res_path = 'D:\\Desktop\\competition\\RFCX\\res'
model_path = 'D:\\Desktop\\competition\\RFCX\\model_save'
tensorboard_path = 'D:\\Desktop\\competition\\RFCX\\tensorboard'
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(res_path):
    os.makedirs(res_path)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
    
num_class = 24
fft = 2048
hop = 512
sr = 48000
length = 10 * sr

data_tp_df=pd.read_csv(os.path.join(data_path, 'train_tp.csv'))
data_fp_df=pd.read_csv(os.path.join(data_path, 'train_fp.csv'))

test_files = sorted(os.listdir(os.path.join(data_path, 'test')))
test_dataset = TestDataset(test_files)
test_dataloader = DataLoader(test_dataset, batch_size=8, sampler=SequentialSampler(test_dataset), shuffle=False, num_workers=6)

with open(os.path.join(model_path, 'resnest_augment_0_02_03_23_36_history.pkl'), 'rb') as f:
    history = pickle.load(f)
model = ResNest().cuda()

if __name__=='__main__':
    folds = []
    for path in history['best_model_path']:
        model.load_state_dict(torch.load(path, map_location= torch.device('cpu')), strict=True)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                a, b, c, d, e = batch.size()
                X = batch.view(a*b, c, d, e).cuda()
                output = model(X)
                pred = F.sigmoid(output).view(a, b, -1).max(dim=1)[0].cpu().detach().numpy()
                preds.append(pred)
        folds.append(np.concatenate(preds, axis=0))
    sub = pd.DataFrame(columns=['recording_id','s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23'])
    sub['recording_id'] = [file.split('.')[0] for file in test_files]
    sub.iloc[:, 1:] = sum(folds)/len(folds)
    time_stamp = '{0:%m_%d_%H_%M}'.format(datetime.datetime.now())
    sub.to_csv(os.path.join(res_path, 'submission_{}.csv'.format(time_stamp)), index=None)
    