import numpy as np
import tensorflow as tf
from cleverhans.dataset import MNIST
import matplotlib.pyplot as plt

mnist = MNIST(train_start=0, train_end=60000,
              test_start=0, test_end=10000)
x_train, y_train = mnist.get_set('train')
x_test, y_test = mnist.get_set('test')
x_test = x_test.reshape(10000,28,28)

fgsm = np.load('adv_fgsm.npy').reshape(10000,28,28)
bim = np.load('adv_bim.npy').reshape(10000,28,28)
mifgsm = np.load('adv_mifgsm.npy').reshape(10000,28,28)

index = np.linspace(1,10,10).astype('int16')

images = [x_test,fgsm,bim,mifgsm]
titles = ["origin",
          "FGSM",
          "BIM",
          "MI-FGSM"]

for i in index:
    plt.figure()
    sub_num = 221
    for j in range(4):
        plt.subplot(sub_num+j)
        plt.title(titles[j],fontsize=8)
        plt.imshow(images[j][i])
    plt.savefig("images_results/num{}.svg".format(i), format="svg", dpi=400)