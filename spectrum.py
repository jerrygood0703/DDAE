import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import scipy
import scipy.io.wavfile as wav
from scipy import signal
import h5py
import librosa

if len(sys.argv) < 4:
    print "Usage: python spectrum.py data.h5 list_noisy list_clean"
    sys.exit(1) 

FRAMESIZE = 512
OVERLAP = 256
FFTSIZE = 512
RATE = 16000
FRAMEWIDTH = 2
FBIN = FRAMESIZE//2+1
data_name = sys.argv[1] #"/mnt/hd-01/user_sylar/MHINTSYPD_100NS/data_257_spectrum.h5"

noisylistpath = sys.argv[2] #"/mnt/hd-01/user_sylar/MHINTSYPD_100NS/trnoisylist"
print "Expected data size: 3000000"
noisydata = np.zeros((3000000,FBIN,FRAMEWIDTH*2+1),dtype=np.float32)
idx = 0
with open(noisylistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print idx
        y,sr=librosa.load(line[:-1],sr=16000)
        D=librosa.stft(y,n_fft=FRAMESIZE,hop_length=OVERLAP,win_length=FFTSIZE,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2) 
        print 'spec.shape' + str(Sxx.shape)
        mean = np.mean(Sxx, axis=1).reshape(FBIN,1)
        std = np.std(Sxx, axis=1).reshape(FBIN,1)+1e-12
        Sxx = (Sxx-mean)/std
        for i in range(FRAMEWIDTH, Sxx.shape[1]-FRAMEWIDTH): # 5 Frmae
            noisydata[idx,:,:] = Sxx[:,i-FRAMEWIDTH:i+FRAMEWIDTH+1] # For Noisy data
            idx = idx + 1

noisydata = noisydata[:idx]
noisydata = np.reshape(noisydata,(idx,-1))
#===================================================================================#
with h5py.File(data_name, 'a') as hf:
    hf.create_dataset('trnoisy', data=noisydata) # For Noisy data
noisdydata = []

cleanlistpath = sys.argv[3] #"/mnt/hd-01/user_sylar/MHINTSYPD_100NS/trcleanlist"
cleandata = np.zeros((3000000,FBIN),dtype=np.float32)
c_idx = 0
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print c_idx
        y,sr=librosa.load(line[:-1],sr=16000)
        D=librosa.stft(y,n_fft=FRAMESIZE,hop_length=OVERLAP,win_length=FFTSIZE,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2) 
        print 'spec.shape' + str(Sxx.shape)
        for i in range(FRAMEWIDTH, Sxx.shape[1]-FRAMEWIDTH): # 5 Frmae        
            cleandata[c_idx,:] = Sxx[:,i] # For Clean data
            c_idx = c_idx + 1

cleandata = cleandata[:c_idx]

with h5py.File(data_name, 'a') as hf:
    hf.create_dataset('trclean', data=cleandata) # For Clean data
