from __future__ import print_function
import librosa
import soundfile as sf
import os
import h5py

sample_rate = 16000

for i in range(50):
    name0 = 'ccmixter2/x/' + str(i) + '.wav'
    audio0, samplerate = sf.read(name0, dtype='float32')
    audio0 = librosa.resample(audio0.T, samplerate, sample_rate)

    name1 = 'ccmixter2/y/' + str(i) + '.wav'
    audio1, samplerate = sf.read(name1, dtype='float32')
    audio1 = librosa.resample(audio1.T, samplerate, sample_rate)
    if not os.path.exists('./ccmixter3/'): os.makedirs('./ccmixter3/')
    h5f = h5py.File('ccmixter3/' + str(i) + '.h5', 'w')
    h5f.create_dataset('x', data=audio0)
    h5f.create_dataset('y', data=audio1)
    h5f.close()