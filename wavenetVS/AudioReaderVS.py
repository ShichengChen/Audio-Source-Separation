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
    

def load_one_audio(directory, sample_rate,trainOrNot=True):
    if(trainOrNot):
        allfiles = [['./vsCorpus/origin_mix.wav','./vsCorpus/origin_vocal.wav']]
    else:
        allfiles = [['./vsCorpus/pred_mix.wav','./vsCorpus/pred_vocal.wav']]
    for filename in allfiles:
        audio0, samplerate = sf.read(filename[0], dtype='float32')
        audio0 = librosa.resample(audio0.T, samplerate, sample_rate)
        audio0 = audio0.reshape(-1, 1)
        
        audio1, samplerate = sf.read(filename[1], dtype='float32')
        audio1 = librosa.resample(audio1.T, samplerate, sample_rate)
        audio1 = audio1.reshape(-1, 1)
        assert(audio0.shape==audio1.shape)
        yield audio0,audio1, filename, 0

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


def trim_silence(audio0,audio1, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio0.size < frame_length:
        frame_length = audio0.size
    energy = librosa.feature.rmse(audio0, frame_length=frame_length//2)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    #print('frame',librosa.core.frames_to_samples(frames))
    # Note: indices can be an empty array, if the whole audio was silence.
    if(len(indices)):return audio0[indices[0]:indices[-1]],audio1[indices[0]:indices[-1]]
    return audio0[0:0],audio1[0:0]


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
        self.trxsample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.trysample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.trxqueue = tf.PaddingFIFOQueue(1,['float32'],shapes=[(None, 1)])
        self.tryqueue = tf.PaddingFIFOQueue(1,['float32'],shapes=[(None, 1)])
        self.trxenqueue = self.trxqueue.enqueue([self.trxsample_placeholder])
        self.tryenqueue = self.tryqueue.enqueue([self.trysample_placeholder])
        
        #self.vxsample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        #self.vysample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        #self.vxqueue = tf.PaddingFIFOQueue(4,['float32'],shapes=[(None, 1)])
        #self.vyqueue = tf.PaddingFIFOQueue(4,['float32'],shapes=[(None, 1)])
        #self.vxenqueue = self.vxqueue.enqueue([self.vxsample_placeholder])
        #self.vyenqueue = self.vyqueue.enqueue([self.vysample_placeholder])
        

        if self.gc_enabled:
            ##TODO trxaudio,tryaudio
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

    def trdequeue(self, num_elements):
        print('trdequeue')
        output = (self.trxqueue.dequeue_many(num_elements),self.tryqueue.dequeue_many(num_elements))
        return output
    '''def vdequeue(self, num_elements):
        print('vdequeue')
        output = (self.vxqueue.dequeue_many(num_elements),self.vyqueue.dequeue_many(num_elements))
        return output'''

    def dequeue_gc(self, num_elements):
        ##TODO trxaudio,tryaudio
        pass
        return self.gc_queue.dequeue_many(num_elements)
    
    
    def valbatch(self):
        stop = False
        trainOrNot=False
        if(trainOrNot):filename = ['./vsCorpus/origin_mix.wav','./vsCorpus/origin_vocal.wav']
        else:filename = ['./vsCorpus/pred_mix.wav','./vsCorpus/pred_vocal.wav']
        print('val',filename)
        audio0, samplerate = sf.read(filename[0], dtype='float32')
        audio0 = librosa.resample(audio0.T, samplerate, self.sample_rate)
        audio0 = audio0.reshape(-1, 1)

        audio1, samplerate = sf.read(filename[1], dtype='float32')
        audio1 = librosa.resample(audio1.T, samplerate, self.sample_rate)
        audio1 = audio1.reshape(-1, 1)
        assert(audio0.shape==audio1.shape)
        vxaudio = np.pad(audio0, [[self.receptive_field, 0], [0, 0]],'constant')
        vyaudio = np.pad(audio1, [[self.receptive_field, 0], [0, 0]],'constant')
        vxaudio=tf.convert_to_tensor(vxaudio, dtype=tf.float32)
        vyaudio=tf.convert_to_tensor(vyaudio, dtype=tf.float32)
        return (vxaudio,vyaudio)
    
    def thread_train(self, sess):
        stop = False
        # Go through the dataset multiple times
        filename = ['./vsCorpus/origin_mix.wav','./vsCorpus/origin_vocal.wav']
        audio0, samplerate = sf.read(filename[0], dtype='float32')
        audio0 = librosa.resample(audio0.T, samplerate, self.sample_rate)
        audio0 = audio0.reshape(-1, 1)[:self.sample_size*4,:]
        audio0 = np.pad(audio0, [[self.receptive_field, 0], [0, 0]],'constant')
        
        audio1, samplerate = sf.read(filename[1], dtype='float32')
        audio1 = librosa.resample(audio1.T, samplerate, self.sample_rate)
        audio1 = audio1.reshape(-1, 1)[:self.sample_size*4,:]
        audio1 = np.pad(audio1, [[self.receptive_field, 0], [0, 0]],'constant')
        assert(audio0.shape==audio1.shape)
        while not stop:
            if self.coord.should_stop():
                stop = True
                break
            if self.silence_threshold is not None:
                # Remove silence
                trxaudio,tryaudio = trim_silence(trxaudio[:, 0],tryaudio[:, 0], self.silence_threshold)
                trxaudio,tryaudio = trxaudio.reshape(-1, 1),tryaudio.reshape(-1, 1)
                if trxaudio.size == 0:
                    print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the trxaudio."
                              .format(filename))

                #print(self.sample_size)   
            if self.sample_size:   ##SAMPLE_SIZE = 100000
                # Cut samples into pieces of size receptive_field +
                # sample_size with receptive_field overlap
                #receptive_field=5117
                lens = self.sample_size+self.receptive_field
                startnum = np.arange(int((len(audio0)-lens)/lens))
                #np.random.shuffle(startnum)
                #print('train',startnum)
                for i in startnum:
                    #print('trx',sess.run(self.trxqueue.size()))
                    #print('try',sess.run(self.tryqueue.size()))
                    #print('tr',filename)
                    trxpiece = audio0[i*lens:(i+1)*lens, :].copy()#+np.random.randn(lens,1)*(1e-4)
                    #trxpiece = np.pad(trxpiece, [[self.receptive_field, 0], [0, 0]],'constant')
                    sess.run(self.trxenqueue,feed_dict={self.trxsample_placeholder: trxpiece})
                     
                    trypiece = audio1[i*lens:(i+1)*lens, :].copy()#+np.random.randn(lens,1)*(1e-4)  
                    #trypiece = np.pad(trypiece, [[self.receptive_field, 0], [0, 0]],'constant')
                    sess.run(self.tryenqueue,feed_dict={self.trysample_placeholder: trypiece})
                        
                    if self.gc_enabled:
                        pass ##TODO trxaudio,tryaudio
                        sess.run(self.gc_enqueue, feed_dict={self.id_placeholder: category_id})
                '''while len(trxaudio) > self.receptive_field:
                    trxpiece = trxaudio[:(self.receptive_field +self.sample_size), :]
                    sess.run(self.trxenqueue,feed_dict={self.trxsample_placeholder: trxpiece})
                    trxaudio = trxaudio[self.sample_size:, :]
                        
                    trypiece = tryaudio[:(self.receptive_field +self.sample_size), :]
                    sess.run(self.tryenqueue,feed_dict={self.trysample_placeholder: trypiece})
                    tryaudio = tryaudio[self.sample_size:, :]'''
                        

            '''else:
                sess.run(self.trxenqueue,feed_dict={self.trxsample_placeholder: trxaudio})
                sess.run(self.tryenqueue,feed_dict={self.trysample_placeholder: tryaudio})
                if self.gc_enabled:
                    ##TODO trxaudio,tryaudio
                    pass
                    sess.run(self.gc_enqueue,
                         feed_dict={self.id_placeholder: category_id})'''

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread0 = threading.Thread(target=self.thread_train, args=(sess,))
            thread0.daemon = True  # Thread will close when parent quits.
            thread0.start()
            self.threads.append(thread0)
        return self.threads
