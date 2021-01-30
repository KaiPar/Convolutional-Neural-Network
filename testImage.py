from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def load_image(filename):
    """
        Load the image
    :param filename: path to the image in string
    :return: image as float 32-bit
    """
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img


def run_example():
    """
        Test the trained Neural Network against input image
    :return: None
    """
    for i in range(10):
        img = load_image(str(i) + '.png')
        model = load_model('final_model.h5')
        digit = model.predict_classes(img)
        print(digit[0])


"""
    To run the Neural Network against any input un-comment the code below and comment the function above
"""

"""
def run_example():
    img = load_image("<your path to the file>")
    model = load_model('final_model.h5")
    digit = model.predict_classes(img)
    print(digit[0])
"""

run_example()