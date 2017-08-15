# DDAE speech enhancement

**Hyper-parameters are not tuned to optimal**

## Prerequisites:
- Keras 1.2
- Tensorflow 1.x as backend
- h5py
- librosa
- scipy

## Getting Started:

Extract spectrogram features:

```sh
python spectrum.py data.h5 list_noisy list_clean
```

Train DDAE using Keras:
```sh
python train_DNN.py data.h5
```

Enhance test wave files using trained model:
```sh
python test_gen_spec.py model.hdf5 list_noisy
```
