from distutils.version import LooseVersion
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout

if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
  from keras.layers import Conv2D
else:
  from keras.layers import Convolution2D

import math

def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):
  """
  Defines the right convolutional layer according to the
  version of Keras that is installed.
  :param filters: (required integer) the dimensionality of the output
                  space (i.e. the number output of filters in the
                  convolution)
  :param kernel_shape: (required tuple or list of 2 integers) specifies
                       the strides of the convolution along the width and
                       height.
  :param padding: (required string) can be either 'valid' (no padding around
                  input or feature map) or 'same' (pad to ensure that the
                  output feature map size is identical to the layer input)
  :param input_shape: (optional) give input shape if this is the first
                      layer of the model
  :return: the Keras layer
  """
def modelA(logits=False, input_ph=None, img_rows=28, img_cols=28,
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

  layers = [conv_2d(nb_filters, (8, 8), (2, 2), "same",
                    input_shape=input_shape),
            Activation('relu'),
            conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
            Activation('relu'),
            conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
            Activation('relu'),
            Flatten(),
            Dense(nb_classes)]

  for layer in layers:
    model.add(layer)

  model.add(Activation('softmax'))

  return model

def modelB(logits=False, input_ph=None, img_rows=28, img_cols=28,
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

  return model

def modelC(logits=False, input_ph=None, img_rows=28, img_cols=28,
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

  model = keras.Sequential([
      keras.layers.Conv2D(input_shape=(28, 28, 1),
                          kernel_size=(3, 3), filters=32, activation='relu'),
      keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
      keras.layers.MaxPool2D(),

      keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
      keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
      keras.layers.MaxPool2D(),

      keras.layers.Flatten(),
      keras.layers.Dense(200, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(200, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
  ])
  return model
