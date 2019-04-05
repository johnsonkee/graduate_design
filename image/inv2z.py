import os, sys
import pickle
import numpy as np
import tensorflow as tf
import argparse
from keras.models import load_model
import tflib.mnist
from mnist_wgan_inv import MnistWganInv

import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_path', type=str, default='./models/model-47999',
                        help='mnist GAN path')
    args = parser.parse_args()

    graph_GAN = tf.Graph()
    with graph_GAN.as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        sess_GAN = tf.Session(config=config)
        model_GAN = MnistWganInv()
        saver_GAN = tf.train.Saver(max_to_keep=100)
        saver_GAN = tf.train.import_meta_graph('{}.meta'.format(args.gan_path))
        saver_GAN.restore(sess_GAN, args.gan_path)

    def inv_fn(x):
        with sess_GAN.as_default():
            with graph_GAN.as_default():
                z_p = sess_GAN.run(model_GAN.invert(x))
        return z_p

    z_adv = []
    x_adv = pickle.load(open("../../server_copyin/adv90.pkl","rb"))

    z_origin = []
    y_origin = []
    _, _, test_data = tflib.mnist.load_data()

    for i in range(len(test_data[0])):
        z_origin.append(inv_fn(test_data[0][i]))
        y_origin.append(test_data[1][i])
    pickle.dump(z_origin,open("../../server_copyin/z_origin_all.pkl","wb"))
    pickle.dump(y_origin,open("../../server_copyin/y_origin_all.pkl","wb"))

