# tf2.0.0
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
mnist = keras.datasets.mnist

(_,_),(test_images,test_labels) = mnist.load_data()

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

generator = keras.models.load_model("../generator.h5")
classifier = keras.models.load_model("../models/FCnet.h5")
"""
# paint adversary
indimg = 6
input = test_images[indimg:indimg+1]
pertubation = generator(input,training=True)

generated_images = input + pertubation
preds = classifier(generated_images)
preds = tf.argmax(preds,axis=1).numpy()[0]
print(preds)

for i in range(1):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("origin,class{}".format(test_labels[indimg]))
    plt.imshow(input.reshape(28,28))
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("generated,class{}".format(preds))
    plt.imshow(generated_images.numpy().reshape(28, 28))
    plt.colorbar()
    plt.show()
"""
input = test_images
pertubation = generator(input,training=True)
generated_images = input + pertubation

print("origin")
classifier.evaluate(test_images,test_labels,verbose=2)
print("adversary")
classifier.evaluate(generated_images,test_labels,verbose=2)
preds = classifier(generated_images)
preds = tf.argmax(preds,axis=1).numpy()

same_number = np.sum(preds == test_labels)
attack_rate = 1 - same_number/len(test_labels)
print("attack_success rate is {}%".format(attack_rate*100))