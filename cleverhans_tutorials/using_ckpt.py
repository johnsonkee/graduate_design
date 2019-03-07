from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from cleverhans.dataset import CIFAR10
import tensorflow as tf
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.utils_tf import model_eval


def restore_model_evaluate(sess):

    saver = tf.train.import_meta_graph('simple_cifar10.ckpt.meta')
    model_file = tf.train.latest_checkpoint('./')
    saver.restore(sess, model_file)
    print('loading adversial models successfully!\n')

def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args={'batch_size': 128})
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

with tf.Session() as sess:
    model1 = ModelAllConvolutional('model1', 10, 64,
                                 input_shape=[32, 32, 3])
    vgg_ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_feat_fc')
    restore_model_evaluate(sess)

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



    preds = model1.get_logits(x)
    sess.run(tf.global_variables_initializer())
    print(tf.get_default_graph().get_operations())
    model = tf.get_default_graph().get_collection("model1")[0]

    do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)
