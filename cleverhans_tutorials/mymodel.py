from distutils.version import LooseVersion
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout

if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
  from keras.layers import Conv2D
else:
  from keras.layers import Convolution2D

import math

def fc_modelB(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
  """
  Defines a CNN model using Keras sequential model
  :param logits: If set to False, returns a Keras model, otherwise will also
                  return logits tensor
  :param input_ph: The TensorFlow tensor for the input
                  (needed if returning logits)
                  ("ph" stands for placeholder but it need not actually be a
                  placeholder)
  :param img_rows: number of row in the image
  :param img_cols: number of columns in the image
  :param channels: number of color channels (e.g., 1 for MNIST)
  :param nb_filters: number of convolutional filters per layer
  :param nb_classes: the number of output classes
  :return:
  """
  model = Sequential()

  # Define the layers successively (convolution layers are version dependent)
  if keras.backend.image_dim_ordering() == 'th':
    input_shape = (channels, img_rows, img_cols)
  else:
    input_shape = (img_rows, img_cols, channels)

  layers = [Flatten(input_shape=input_shape),
            Dense(nb_filters),
            Activation('relu'),
            Dense(nb_filters * 2),
            Activation('relu'),
            Dense(nb_filters * 4),
            Activation('relu'),
            Dropout(0.2),
            Dense(nb_classes)]

  for layer in layers:
    model.add(layer)

  if logits:
    logits_tensor = model(input_ph)
  model.add(Activation('softmax'))

  if logits:
    return model, logits_tensor
  else:
    return model

