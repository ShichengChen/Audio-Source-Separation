from network import *
from data_utils import *
import hyperparams as hp
import librosa
import soundfile as sf

class Graph:
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, hp.timestep, 1], name='X')

            output = network(self.x, use_mulaw=hp.use_mulaw)

            if hp.use_mulaw:
                self.prediction = mu_law_decode(tf.argmax(output, axis=2))
            else:
                self.prediction = tf.squeeze(output, -1)

def main():

    g = Graph()
    
    filename = ['./vsCorpus/pred_mix.wav','./vsCorpus/pred_vocal.wav']
    mixture, samplerate = sf.read(filename[0], dtype='float32')
    mixture = librosa.resample(audio0.T, samplerate, hp.sample_rate)
    mixture = audio0.reshape(-1)
    
    mixture_len = len(mixture) // hp.timestep
    print (mixture_len)
    mixture = np.expand_dims(mixture[:mixture_len * hp.timestep].reshape([-1,hp.timestep]),-1)

    with g.graph.as_default():

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(hp.save_dir))
            print "restore successfully!"

            outputs = []
            for part in mixture:
                part = np.expand_dims(part, axis=0)
                output = sess.run(g.prediction, feed_dict={g.x:part})
                np.squeeze(output, axis=0)
                outputs.append(output)

            result = np.vstack(outputs).reshape(-1)
            sf.write('./vsCorpus/result.wav',result,hp.sample_rate)

if __name__ == '__main__':
    main()

