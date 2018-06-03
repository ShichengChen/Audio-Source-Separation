import os
import numpy as np
import librosa
import hyperparams as hp

def load_data():
    mixtures, vocals = list(), list()
    for path, subdirs, files in os.walk('./data/DSD100/Mixtures/Dev'):
        for name in [f for f in files if f.endswith(".wav")]:
            # a = librosa.load(os.path.join(path, name), sr=44100)[0].shape
            mixtures.append(os.path.join(path, name))

    for path, subdirs, files in os.walk('./data/DSD100/Sources/Dev'):
        for subdir in subdirs:
            vocal = os.path.join(os.path.join(path, subdir), "vocals.wav")
            vocals.append(vocal)

    num_wavs = len(mixtures)

    return mixtures, vocals, num_wavs

def get_rawwave(_input):
    return librosa.load(_input, sr=hp.sample_rate)

def make_rawdata(is_training=True, name="data"):

    m, v, n = load_data()
    arrays = []
    arrays_2 = []
    for i, j in zip(m, v):
        data = get_rawwave(i)[0]
        lens = len(data) // hp.timestep
        arrays.append(np.expand_dims(np.reshape(data[:hp.timestep * lens], [-1, hp.timestep]), -1))

        if is_training:
            data_2 = get_rawwave(j)[0]
            arrays_2.append(np.expand_dims(np.reshape(data_2[:hp.timestep * lens], [-1, hp.timestep]), -1))
    print np.vstack(arrays).shape
    np.save("./data/mixtures_%s.npy" % name, np.vstack(arrays))
    if is_training:
        np.save("./data/vocals.npy", np.vstack(arrays_2))

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
