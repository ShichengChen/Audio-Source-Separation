from network import *
from data_utils import *
import hyperparams as hp


class Graph:
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, hp.timestep, 1], name='X')
            self.y = tf.placeholder(tf.float32, [None, hp.timestep, 1], name='Y')
            if hp.use_mulaw:
                label = mu_law_encode(self.y)
            else:
                label = self.y

            output = network(self.x, use_mulaw=hp.use_mulaw)

            if hp.use_mulaw:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label))
            else:
                self.loss = tf.reduce_mean(tf.abs(output - label))

            self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

            tf.summary.scalar("loss", self.loss)
            self.merged = tf.summary.merge_all()

def main():
    mixture = np.load('./data/mixtures_data.npy')
    vocals = np.load('./data/vocals.npy')

    num_batch = len(mixture) // hp.batch_size

    g = Graph()

    with g.graph.as_default():
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allocator_type = 'BFC'
        # config.gpu_options.per_process_gpu_memory_fraction = 0.80
        # config.gpu_options.allow_growth = True

        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in xrange(hp.num_epochs):

                mixture, vocals = dataset_shuffling(mixture, vocals)
                for i in range(num_batch):
                    batch_X, batch_Y = get_batch(mixture, vocals, i, hp.batch_size)
                    sess.run(g.train_op, feed_dict={g.x:batch_X, g.y:batch_Y})

                    if i % 100 == 0:
                        print "step %d, CEloss:%.4f" %(i,sess.run(g.loss, feed_dict={g.x:batch_X, g.y:batch_Y}))
                        saver.save(sess, hp.save_dir+"/model_%d.ckpt" % (epoch*num_batch + i))
if __name__ == '__main__':
    main()

