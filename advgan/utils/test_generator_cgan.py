import tensorflow.keras as keras
import tensorflow as tf
from matplotlib import pyplot as plt
mnist = keras.datasets.mnist

(_,_),(test_images,test_labels) = mnist.load_data()

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
test_images = test_images.reshape(-1, 784)
test_labels_onehot = keras.utils.to_categorical(test_labels).reshape(-1, 10)

generator = keras.models.load_model("../generator_cgan.h5")
classifier = keras.models.load_model("../models/FCnet.h5")


input = tf.concat([test_images,test_labels_onehot],axis=1)
perturbation = generator(input,training=True)
generated_images = tf.reshape(test_images,[-1,28,28,1]) + perturbation

test_images = test_images.reshape(-1,28,28,1)

print("origin")
classifier.evaluate(test_images,test_labels,verbose=2)
print("adversary")
classifier.evaluate(generated_images,test_labels,verbose=2)
