import tensorflow as tf

def resize_img(image, label):
    return tf.image.resize(image,IMG_SIZE),label