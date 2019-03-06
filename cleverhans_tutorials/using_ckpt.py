
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.dataset import CIFAR10
from cleverhans.loss import CrossEntropy
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval

model_path = sys.path[0] + '/simple_mnist.ckpt'


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('simple_cifar10.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint("."))

    data = CIFAR10(train_start=0, train_end=60000,
                   test_start=0, test_end=10000)
    dataset_size = data.x_train.shape[0]
    dataset_train = data.to_tensorflow()[0]
    dataset_train = dataset_train.map(
        lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
    dataset_train = dataset_train.batch(128)
    dataset_train = dataset_train.prefetch(16)
    x_train, y_train = data.get_set('train')
    x_test, y_test = data.get_set('test')

    # Use Image Parameters
    img_rows, img_cols, nchannels = x_test.shape[1:4]
    nb_classes = y_test.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    eval_params = {'batch_size': 128}
    def do_eval(preds, x_set, y_set, report_key, is_adv=None):
        acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
        if is_adv is None:
            report_text = None
        elif is_adv:
            report_text = 'adversarial'
        else:
            report_text = 'legitimate'
        if report_text:
            print('Test accuracy on %s examples: %0.4f' % (report_text, acc))


    model = ModelAllConvolutional('model1', nb_classes, nb_filters=64,
                                  input_shape=[32, 32, 3])
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    def evaluate():
        do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)
    evaluate()

