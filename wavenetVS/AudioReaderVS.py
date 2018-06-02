import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_one_audio(directory, sample_rate):
    allfiles = [['./vocalSeparation/origin_mix.wav','./vocalSeparation/origin_vocal.wav']]
    for filename in allfiles:
        print(filename[0],filename[1])
        audio0, samplerate = sf.read(filename[0], dtype='float32')
        audio0 = librosa.resample(audio0.T, samplerate, sample_rate)
        audio1, samplerate = sf.read(filename[1], dtype='float32')
        audio1 = librosa.resample(audio0.T, samplerate, sample_rate)
        audio0 = audio0.reshape(-1, 1)
        audio1 = audio1.reshape(-1, 1)
        yield audio0,audio1, filename[0], 0

def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        #print(filename)
        ids = id_reg_exp.findall(filename)
        #print(ids)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        #print(librosa.load(filename, sr=sample_rate, mono=True)[0].shape) #(65584,) 16000
        #print(librosa.load(filename, mono=True)[0].shape,librosa.load(filename, mono=True)[1])  #(90383,) 22050
        #(65584,) 16000 ((65584,) / 16000 == (90383,) 22050)True
        audio = audio.reshape(-1, 1)
        #print(filename, category_id)
        yield audio, filename, category_id


def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    #print('frame',librosa.core.frames_to_samples(frames))
    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        self.xsample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.ysample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.xqueue = tf.PaddingFIFOQueue(queue_size, #queue_size=32
                                         ['float32'],shapes=[(None, 1)])
        self.yqueue = tf.PaddingFIFOQueue(queue_size, #queue_size=32
                                         ['float32'],shapes=[(None, 1)])
        self.xenqueue = self.xqueue.enqueue([self.xsample_placeholder])
        self.yenqueue = self.yqueue.enqueue([self.ysample_placeholder])

        if self.gc_enabled:
            ##TODO xaudio,yaudio
            pass
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            ##TODO xaudio,yaudio
            pass
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        output = (self.xqueue.dequeue_many(num_elements),self.yqueue.dequeue_many(num_elements))
        return output

    def dequeue_gc(self, num_elements):
        ##TODO xaudio,yaudio
        pass
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_one_audio(self.audio_dir, self.sample_rate)  ##"sample_rate": 16000,
            #iterator = load_generic_audio(self.audio_dir, self.sample_rate)  ##"sample_rate": 16000,
            for xaudio,yaudio, filename, category_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    ##TODO xaudio,yaudio
                    pass
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))

                xaudio = np.pad(xaudio, [[self.receptive_field, 0], [0, 0]],'constant')
                yaudio = np.pad(yaudio, [[self.receptive_field, 0], [0, 0]],'constant')
                #print(self.sample_size)   
                if self.sample_size:   ##SAMPLE_SIZE = 100000
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    #receptive_field=5117
                    while len(xaudio) > self.receptive_field:
                        xpiece = xaudio[:(self.receptive_field +self.sample_size), :]
                        sess.run(self.xenqueue,feed_dict={self.xsample_placeholder: xpiece})
                        xaudio = xaudio[self.sample_size:, :]
                        
                        ypiece = yaudio[:(self.receptive_field +self.sample_size), :]
                        sess.run(self.yenqueue,feed_dict={self.ysample_placeholder: ypiece})
                        yaudio = yaudio[self.sample_size:, :]
                        
                        if self.gc_enabled:
                            ##TODO xaudio,yaudio
                            pass
                            sess.run(self.gc_enqueue, feed_dict={
                                self.id_placeholder: category_id})
                else:
                    sess.run(self.xenqueue,feed_dict={self.xsample_placeholder: xaudio})
                    sess.run(self.yenqueue,feed_dict={self.ysample_placeholder: yaudio})
                    if self.gc_enabled:
                        ##TODO xaudio,yaudio
                        pass
                        sess.run(self.gc_enqueue,
                                 feed_dict={self.id_placeholder: category_id})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
