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

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
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

import albumentations as A
import albumentations.pytorch.transforms as T

import matplotlib.pyplot as plt

SEED = 42
DATA_PATH = "C:\\workspace\\data\\audio\\birdclef-2022"
AUDIO_PATH = 'C:\\workspace\\data\\audio\\birdclef-2022\\train_audio'
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
NUM_WORKERS = 4
CLASSES = sorted(os.listdir(AUDIO_PATH))
NUM_CLASSES = len(CLASSES)

class AudioParams:
    """
    Parameters used for the audio data
    """
    sr = 32000
    duration = 5
    # Melspectrogram
    n_mels = 224
    fmin = 20
    fmax = 16000


train = pd.read_csv(DATA_PATH + '/train_metadata.csv')
train["file_path"] = AUDIO_PATH + '/' + train['filename']
paths = train["file_path"].values

Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for n, (trn_index, val_index) in enumerate(Fold.split(train, train['primary_label'])):
    train.loc[val_index, 'kfold'] = int(n)
train['kfold'] = train['kfold'].astype(int)

train.to_csv('train_folds.csv', index=False)

print(train.shape)
train.head()

def compute_melspec(y, params):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = librosa.feature.melspectrogram(
        y=y, sr=params.sr, n_mels=params.n_mels, fmin=params.fmin, fmax=params.fmax,
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec


def crop_or_pad(y, length, sr, train=True, probs=None):
    """
    Crops an array to a chosen length
    Arguments:
        y {1D np array} -- Array to crop
        length {int} -- Length of the crop
        sr {int} -- Sampling rate
    Keyword Arguments:
        train {bool} -- Whether we are at train time. If so, crop randomly, else return the beginning of y (default: {True})
        probs {None or numpy array} -- Probabilities to use to chose where to crop (default: {None})
    Returns:
        1D np array -- Cropped array
    """
    if len(y) <= length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    else:
        if not train:
            start = 0
        elif probs is None:
            start = np.random.randint(len(y) - length)
        else:
            start = (
                    np.random.choice(np.arange(len(probs)), p=probs) + np.random.random()
            )
            start = int(sr * (start))

        y = y[start: start + length]

    return y.astype(np.float32)


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Converts a one channel array to a 3 channel one in [0, 255]
    Arguments:
        X {numpy array [H x W]} -- 2D array to convert
    Keyword Arguments:
        eps {float} -- To avoid dividing by 0 (default: {1e-6})
        mean {None or np array} -- Mean for normalization (default: {None})
        std {None or np array} -- Std for normalization (default: {None})
    Returns:
        numpy array [3 x H x W] -- RGB numpy array
    """
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
# original

path = train['file_path'][0]
y, sr = sf.read(path, always_2d=True)
y = np.mean(y, 1)

X = compute_melspec(y, AudioParams)
X = mono_to_color(X)
X = X.astype(np.uint8)

plt.imshow(X)
plt.show()
# 5 sec cropped

path = train['file_path'][0]
y, sr = sf.read(path, always_2d=True)
y = np.mean(y, 1)
y = crop_or_pad(y, AudioParams.duration * AudioParams.sr, sr=AudioParams.sr, train=True, probs=None)

X = compute_melspec(y, AudioParams)
X = mono_to_color(X)
X = X.astype(np.uint8)

plt.imshow(X)
plt.show()
def Audio_to_Image(path, params):
    y, sr = sf.read(path, always_2d=True)
    y = np.mean(y, 1) # there is (X, 2) array
    y = crop_or_pad(y, params.duration * params.sr, sr=params.sr, train=True, probs=None)
    image = compute_melspec(y, params)
    image = mono_to_color(image)
    image = image.astype(np.uint8)
    return image

def save_(path):
    save_path = "../working/" + "/".join(path.split('/')[-2:])
    np.save(save_path, Audio_to_Image(path, AudioParams))
# Parallel Execution
NUM_WORKERS = 4
for dir_ in CLASSES:
    _ = os.makedirs(dir_, exist_ok=True)
_ = Parallel(n_jobs=NUM_WORKERS)(delayed(save_)(AUDIO_PATH) for AUDIO_PATH in tqdm(paths))