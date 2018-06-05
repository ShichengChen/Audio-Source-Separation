
# coding: utf-8

# In[1]:


from network import *
from data_utils import *
import hyperparams as hp


# In[2]:

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



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
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label))
            else:
                self.loss = tf.reduce_mean(tf.abs(output - label))

            self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

            tf.summary.scalar("loss", self.loss)
            self.merged = tf.summary.merge_all()





# In[3]:


def main():
    mixture = np.load('mixtures.npy')
    vocals = np.load('vocals.npy')

    num_batch = len(mixture) // hp.batch_size

    g = Graph()

    with g.graph.as_default():
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allocator_type = 'BFC'
        # config.gpu_options.per_process_gpu_memory_fraction = 0.80
        # config.gpu_options.allow_growth = True

        saver = tf.train.Saver()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(hp.num_epochs):

                mixture, vocals = dataset_shuffling(mixture, vocals)
                for i in range(num_batch):
                    batch_X, batch_Y = get_batch(mixture, vocals, i, hp.batch_size)
                    sess.run(g.train_op, feed_dict={g.x:batch_X, g.y:batch_Y})
                    
                    if i % 100 == 0:
                        print ("step %d, CEloss:%.4f" %(i,sess.run(g.loss, feed_dict={g.x:batch_X, g.y:batch_Y})))
                        saver.save(sess, hp.save_dir+"/model_%d.ckpt" % (epoch*num_batch + i))
if __name__ == '__main__':
    main()

