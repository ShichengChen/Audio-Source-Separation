
# coding: utf-8

# In[16]:


import os
import numpy as np
import librosa
import hyperparams as hp
import soundfile as sf


# In[15]:


arrays=[]
data = np.arange(10)
timestep = 3
lens = 3
print((np.reshape(data[:timestep * lens],[-1, 3]).shape))
print((np.expand_dims(np.reshape(data[:timestep * lens],[-1, 3]),-1)).shape)
arrays.append(np.expand_dims(np.reshape(data[:timestep * lens],[-1, 3]),-1))
arrays.append(np.expand_dims(np.reshape(data[:timestep * lens],[-1, 3]),-1))
np.vstack(arrays).shape


# In[19]:


def get_rawwave(_input):
    audio0, samplerate = sf.read(_input, dtype='float32')
    audio0 = librosa.resample(audio0.T, samplerate, hp.sample_rate)
    audio0 = audio0.reshape(-1)
    return audio0

def make_rawdata(is_training=True, name="data"):

    i = '../vsCorpus/origin_mix.wav'
    j = '../vsCorpus/origin_vocal.wav'
    arrays = []
    arrays_2 = []
    data = get_rawwave(i)
    print(data.shape)
    lens = len(data) // hp.timestep
    arrays.append(np.expand_dims(np.reshape(data[:hp.timestep * lens], 
                                                [-1, hp.timestep]), -1))

    data_2 = get_rawwave(j)
    arrays_2.append(np.expand_dims(np.reshape(data_2[:hp.timestep * lens], 
                                                      [-1, hp.timestep]), -1))
    print (np.vstack(arrays).shape)
    np.save("./mixtures.npy", np.vstack(arrays))

    np.save("./vocals.npy", np.vstack(arrays_2))

def dataset_shuffling(x, y):
    shuffled_idx = np.arange(len(y))
    np.random.shuffle(shuffled_idx)
    return x[shuffled_idx, :], y[shuffled_idx, :]

def get_batch(x, y, curr_index, batch_size):
    batch_x = x[curr_index * batch_size: (curr_index+1)*batch_size]
    batch_y = y[curr_index * batch_size: (curr_index+1)*batch_size]
    return batch_x, batch_y

if __name__ == '__main__':
    make_rawdata(is_training=hp.is_training)

