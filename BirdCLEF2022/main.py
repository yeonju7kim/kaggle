import PIL.Image
from PIL import Image
import cv2
import audioread
import logging
import gc
import os
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import random
import time
import warnings
import cProfile
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
from torch import optim, tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from contextlib import contextmanager
from joblib import Parallel, delayed
from pathlib import Path
from typing import Optional
from sklearn.model_selection import StratifiedKFold, GroupKFold

from albumentations.core.transforms_interface import ImageOnlyTransform
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm
from torch import cuda
import albumentations as A
import albumentations.pytorch.transforms as T

import matplotlib.pyplot as plt
# from pytorch_pretrained_vit import ViT
# model = ViT('B_16_imagenet1k', pretrained=True)
from torch.utils.data import DataLoader, Dataset

SEED = 42
DATA_PATH = "C:\\workspace\\data\\audio\\birdclef-2022"
AUDIO_PATH = 'C:\\workspace\\data\\audio\\birdclef-2022\\train_audio'
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
NUM_WORKERS = 4
CLASSES = sorted(os.listdir(AUDIO_PATH))
NUM_CLASSES = len(CLASSES)
EPOCH = 100
KFOLDNUM = 5
device = 'cuda' if cuda.is_available() else 'cpu'
BATCHSIZE = 256

class AudioParams:
    """
    Parameters used for the audio data
    """
    sr = 32000 # sampling rate
    duration = 5 # duration
    # Melspectrogram
    n_mels = 224
    fmin = 20
    fmax = 16000

def compute_melspec(y, params):
    # 소리를 melspectrogram으로 바꾸는 것
    melspec = librosa.feature.melspectrogram(
        y=y, sr=params.sr, n_mels=params.n_mels, fmin=params.fmin, fmax=params.fmax,
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    # 흑백에서 컬러로 바꾸고, 정규화
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def read_sound_list(path_list):
    return [read_sound(path) for path in path_list]

def read_sound(path):
    y, sr = sf.read(path, always_2d=True)
    y = np.mean(y, 1)
    return y

def sound_to_image(sound):
    image = compute_melspec(sound, AudioParams)
    image = mono_to_color(image)
    image = image.astype(np.uint8)
    return image

def imshow(img):
    plt.imshow(img)
    plt.show()

train = pd.read_csv(DATA_PATH + '/train_metadata.csv')
train["file_path"] = AUDIO_PATH + '/' + train['filename']

for path, label in zip(train["file_path"], train['primary_label']):
    saved_path = f'./sound_img/{label}/'
    if os.path.exists(saved_path) == False:
        os.mkdir(saved_path)
    sound_img = Image.fromarray(sound_to_image(read_sound(path)))
    sound_img.save(saved_path+path.split('/')[-1].split('.')[0]+'.jpg')
#
# train_data = read_sound_list(train["file_path"])
# paths = train["file_path"].values
# all_label = set(train['primary_label'])
# Fold = StratifiedKFold(n_splits=KFOLDNUM, shuffle=True, random_state=SEED)
#
# class BirdDataset(Dataset):
#     def __init__(self, sound_list, label_list, num_label):
#         self.sound_list = sound_list
#         self.label_list = label_list
#         self.label_num = num_label
#
#     def __len__(self):
#         return len(self.label_list)
#
#     def __getitem__(self, idx):
#         label_onehot = np.zeros(self.label_num)
#         label_onehot[self.label_list[idx]] = 1
#         return self.sound_list[idx], self.label_list[idx], label_onehot
#
# train_dataloader_list = []
# valid_dataloader_list = []
#
# for n, (trn_index, val_index) in enumerate(Fold.split(train, train['primary_label'])):
#     train_dataset = BirdDataset(read_sound_list(train["file_path"][trn_index]), train['primary_label'][trn_index], len(all_label))
#     valid_dataset = BirdDataset(read_sound_list(train["file_path"][val_index]), train['primary_label'][val_index], len(all_label))
#     train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
#     valid_dataloader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=True)
#     train_dataloader_list.append(train_dataloader)
#     valid_dataloader_list.append(valid_dataloader)
#
#     # train.loc[val_index, 'kfold'] = int(n)
# # train['kfold'] = train['kfold'].astype(int)
#
#
#
# train.to_csv('train_folds.csv', index=False)
#
# print(train.shape)
# train.head()
#
# path = train['file_path'][0]
#
# read_sound = read_sound(path)
#
# # from pytorch_pretrained_vit import ViT
# # model = ViT('B_16_imagenet1k', pretrained=True)
#
# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
#
# optimizer = optim.SGD(efficientnet.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)
# criterion = nn.CrossEntropyLoss()
#
# efficientnet.train()
# for epoch in range(EPOCH):
#     train_loss = 0
#     correct = 0
#     total = 0
#     for fold in range(KFOLDNUM):
#         for batch_idx, (inputs, targets, targets_onehot) in enumerate(train_dataloader_list[fold]):
#             inputs, targets, targets_onehot = inputs.to(device), targets.to(device), tensor(targets_onehot).to(device)
#             optimizer.zero_grad()
#
#             outputs = efficientnet(inputs.float())
#             loss = criterion(outputs, targets_onehot)
#             loss.backward()
#
#             optimizer.step()
#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#
#             total += targets.size(0)
#             current_correct = predicted.eq(targets).sum().item()
#             correct += current_correct
#
#     print(f'epoch : {epoch}, loss :')
