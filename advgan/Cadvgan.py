# tensorflow2.0.0a0
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import time
import pdb
from IPython import display
import pandas


# the input is picture x
def make_generator_model():
    """
    :return: tensor:[None,784]
    """
    model = tf.keras.Sequential()
    # the images has been flatten to 784, and the one-hot labels are added.
    model.add(layers.Dense(28*28, use_bias=False, input_shape=(794,)))
    model.add(layers.Reshape((28,28,1)))
    assert model.output_shape == (None, 28, 28, 1)  # Note: None is the batch size

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)  # Note:None is the size of batch
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', use_bias=False,
                            input_shape=[28, 28, 1]))
    assert model.output_shape == (None, 7, 7, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    # the images has been flatten to 784, and the one-hot labels are adde d.
    model.add(layers.Dense(28*28, use_bias=False, input_shape=(794,)))
    model.add(layers.Reshape((28,28,1)))
    assert model.output_shape == (None, 28, 28, 1)  # Note: None is the batch size

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def classifier_loss(preds, target, is_targeted=False):
    # if it is targeted attack
    # 有一种做法是把target(integer)转化成onehot
    # normal loss:
    """
    if is_targeted:
        return cross_entropy(preds, target)
    return -cross_entropy(preds, target)
    """
    # C&W loss
    real = tf.reduce_sum(target * preds, 1)
    other = tf.reduce_max((1 - target) * preds - (target * 10000), 1)
    if is_targeted:
        return tf.reduce_sum(tf.maximum(0.0, other - real))
    return tf.reduce_sum(tf.maximum(0.0, real - other))

def perturb_loss(perturbation, thresh=0.3):
    zeros = tf.zeros((tf.shape(perturbation)[0]))
    # norm-2
    return tf.reduce_mean(
        tf.maximum(zeros, tf.norm(tf.reshape(perturbation, (tf.shape(perturbation)[0], -1)), axis=1) - thresh))


def total_loss(f_loss, gan_loss, perturb_loss, alpha=1.0, beta=5.0):
    """
    """
    total_loss = f_loss + alpha * gan_loss + beta * perturb_loss

    return total_loss

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, labels, alpha, beta):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # pdb.set_trace()
        perturbation = generator(tf.concat([images,labels],1), training=True)
        generated_images = images + tf.reshape(perturbation,[-1,784])

        real_output = discriminator(tf.concat([images,labels],1), training=True)
        fake_output = discriminator(tf.concat([generated_images,labels],1), training=True)
        preds = classifier(tf.reshape(generated_images,[-1,28,28,1]))

        class_loss = classifier_loss(preds, labels, is_targeted=False)
        gen_loss = generator_loss(fake_output)
        pert_loss = perturb_loss(perturbation, thresh=0.3)

        all_loss = total_loss(class_loss, gen_loss, pert_loss, alpha=alpha, beta=beta)
        disc_loss = discriminator_loss(real_output, fake_output)


    gradients_of_generator = gen_tape.gradient(all_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return all_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()

def train(dataset, labels, epochs):
    record_loss = []
    for epoch in range(epochs):
        start = time.time()

        for image_batch, label_batch in zip(dataset, labels):
            all_loss, disc_loss = train_step(image_batch, label_batch,alpha=1,beta=2)
        print("gen_loss:{},disc_loss:{}".format(all_loss.numpy(),disc_loss.numpy()))
        record_loss.append([epoch+1,all_loss.numpy(),disc_loss.numpy()])
        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        # generate_and_save_images(generator,epoch + 1,test_images[1])

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generator.save("generator_cgan_a1_b2.h5")
            discriminator.save("discriminator_cgan_a1_b2.h5")

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    df = pandas.DataFrame(record_loss,index=False,columns=["epoch","gen_loss","disc_loss"])
    if not os.path.exists("log"):
        os.makedirs("log")
    df.to_csv("log/Cadvgan_loss.csv")
    display.clear_output(wait=True)
    # Generate after the final epoch

    # generate_and_save_images(generator,epochs)

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # use the gpu memory depending on our needs
    tf.config.gpu.set_per_process_memory_growth(enabled=True)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    train_images = train_images.reshape(-1, 784)
    train_labels = keras.utils.to_categorical(train_labels).reshape(-1, 10)

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    # Batch and shuffle the data
    # no shuffle
    train_images_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE)
    # shuffle
    # train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    use_model = False
    if use_model:
        generator = keras.models.load_model("generator_cgan.h5")
        discriminator = keras.models.load_model("discriminator_cgan.h5")
    else:
        generator = make_generator_model()
        discriminator = make_discriminator_model()

    classifier_path = "./models/CNN_mnist.h5"
    classifier = keras.models.load_model(classifier_path)
    classifier.evaluate(test_images, test_labels, verbose=1)

    # https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py Line163
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4,beta_1 = 0.5,beta_2 = 0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4,beta_1 = 0.5,beta_2 = 0.9)

    checkpoint_dir = './training_checkpoints_cgan'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    EPOCHS = 1500
    train(train_images_dataset,train_labels_dataset,EPOCHS)
