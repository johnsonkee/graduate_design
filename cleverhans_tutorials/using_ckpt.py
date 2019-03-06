import tensorflow as tf
import sys

model_path = sys.path[0] + '/simple_mnist.ckpt'


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('simple_cifar10.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint("."))