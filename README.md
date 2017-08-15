# DDAE speech enhancement

Extract spectrogram features:

```
python spectrum.py data.h5 list_noisy list_clean
```

Train DDAE using Keras:
```
python train_DNN.py data.h5
```

Enhance test wave files using trained model:
```
python test_gen_spec.py model.hdf5 list_noisy
```
