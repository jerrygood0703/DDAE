from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adagrad, RMSprop
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.utils.io_utils import HDF5Matrix
from scipy import signal
import scipy.io
import scipy.io.wavfile as wav
import numpy as np
import h5py
import librosa
import sys
import os

def make_spectrum_phase(y, FRAMESIZE, OVERLAP, FFTSIZE):
    D=librosa.stft(y,n_fft=FRAMESIZE,hop_length=OVERLAP,win_length=FFTSIZE,window=scipy.signal.hamming)
    Sxx = np.log10(abs(D)**2) 
    phase = np.exp(1j * np.angle(D))
    mean = np.mean(Sxx, axis=1).reshape((257,1))
    std = np.std(Sxx, axis=1).reshape((257,1))+1e-12
    Sxx = (Sxx-mean)/std  
    return Sxx, phase, mean, std

def recons_spec_phase(Sxx_r, phase):
    Sxx_r = np.sqrt(10**Sxx_r)
    R = np.multiply(Sxx_r , phase)
    result = librosa.istft(R,
                     hop_length=256,
                     win_length=512,
                     window=scipy.signal.hamming)
    return result

if len(sys.argv) < 3:
    print "Usage: python test_gen_spec.py model.hdf5 list_noisy"
    sys.exit(1) 

model=load_model(sys.argv[1]) #"weights/DNN_spec_20160425v2.hdf5"
FRAMESIZE = 512
OVERLAP = 256
FFTSIZE = 512
RATE = 16000
FRAMEWIDTH = 2
FBIN = FRAMESIZE//2+1
noisylistpath = sys.argv[2]

with open(noisylistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print filename
        y,sr=librosa.load(line[:-1],sr=16000)
        training_data = np.empty((10000, FBIN, FRAMEWIDTH*2+1)) # For Noisy data

        Sxx, phase, mean, std = make_spectrum_phase(y, FRAMESIZE, OVERLAP, FFTSIZE)
        idx = 0     
        for i in range(FRAMEWIDTH, Sxx.shape[1]-FRAMEWIDTH): # 5 Frmae
            training_data[idx,:,:] = Sxx[:,i-FRAMEWIDTH:i+FRAMEWIDTH+1] # For Noisy data
            idx = idx + 1

        X_train = training_data[:idx]
        X_train = np.reshape(X_train,(idx,-1))
        predict = model.predict(X_train)
        count=0
        for i in range(FRAMEWIDTH, Sxx.shape[1]-FRAMEWIDTH):
            Sxx[:,i] = predict[count]
            count+=1
        # # The un-enhanced part of spec should be un-normalized
        Sxx[:, :FRAMEWIDTH] = (Sxx[:, :FRAMEWIDTH] * std) + mean
        Sxx[:, -FRAMEWIDTH:] = (Sxx[:, -FRAMEWIDTH:] * std) + mean    

        recons_y = recons_spec_phase(Sxx, phase)
        output = librosa.util.fix_length(recons_y, y.shape[0])
        wav.write(os.path.join("enhanced",filename),16000,np.int16(output*32767))