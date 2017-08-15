# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:42:41 2016

@author: Jason
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, AtrousConv2D, ZeroPadding2D
from keras.layers.local import LocallyConnected2D
from keras.optimizers import *
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import *
import time  
import numpy as np
import h5py
import sys

if len(sys.argv) < 2:
    print "Usage: python train_DNN.py data.h5"
    sys.exit(1) 

date = "20160508"
FRAMESIZE = 512
FRAMEWIDTH = 2
FBIN = FRAMESIZE//2+1
input_dim = FBIN*(FRAMEWIDTH*2+1)

BATCHSIZE = 200
EPOCH = 30
print 'model building...'
model = Sequential()

# model.add(Reshape((2048,), input_shape=(1,2048,1)))

model.add(Dense(2048, input_shape=(1285,)))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(2048))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(2048))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(2048))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(2048))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(257))
#model.add(Activation('tanh'))
model.summary()


adam=Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam)

data_path = sys.argv[1] #"/mnt/hd-01/user_sylar/MHINTSYPD_100NS/data_257_spectrum.h5"
print 'data loading...'
X_train = HDF5Matrix(data_path,"trnoisy")
y_train = HDF5Matrix(data_path,"trclean")


checkpointer = ModelCheckpoint(
						filepath="model.hdf5",
						monitor="loss",
						mode="min",
						verbose=0,
						save_best_only=True)
print 'training...'    
hist=model.fit(X_train, y_train, epochs=EPOCH, batch_size=BATCHSIZE, verbose=1,shuffle="batch", callbacks=[checkpointer])
