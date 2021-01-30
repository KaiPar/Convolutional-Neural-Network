"""
    Load sample images from the keras library
"""

from keras.datasets import mnist
from matplotlib import pyplot

(trainX, trainY), (testX, testY) = mnist.load_data()

print("Train: X=%s y=%s" % (trainX.shape, trainY.shape))
print("Test: X=%s y=%s" % (testX.shape, testY.shape))

for i in range(9):
    pyplot.subplot(330 + 1 + i)  # Make a subplot
    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))  # Load the image from MNIST and use pyplot to show the image

pyplot.show()
