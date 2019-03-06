import tensorflow as tf
import sys

model_path = sys.path[0] + '/simple_mnist.ckpt'
saver = tf.train.Saver()


with tf.Session() as sess:
    load_path = saver.restore(sess, model_path)
    print ("[+] Model restored from %s" % load_path)