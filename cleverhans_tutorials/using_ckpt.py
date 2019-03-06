
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import tensorflow as tf
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional

model_path = sys.path[0] + '/simple_mnist.ckpt'

def restore_model_evaluate(sess):

    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint('./')
    saver.restore(sess, model_file)
    print('loading adversial models successfully!\n')

with tf.Session() as sess:
    model = ModelAllConvolutional('model2', 10, 64,
                                 input_shape=[32, 32, 3])
    restore_model_evaluate(sess)
