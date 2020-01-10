import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf
from layer_resize_images import ResizeImages

def build_one_image_encoder(size):
    ni = Input((None, size, size, 3))
    nn = Conv2d(64, (5, 5), (2, 2), tf.nn.relu)(ni)
    nn = Conv2d(64, (5, 5), (1, 1), tf.nn.relu)(nn)
    nn = Conv2d(64, (5, 5), (2, 2), tf.nn.relu)(nn)
    nn = Conv2d(32, (5, 5), (1, 1))(nn)
    return tl.models.Model(inputs = ni, outputs = nn)

# 四种不同的scale
scaling_factors = [2.0, 1.0, 0.5, 0.25]

def build_image_encoder():
    ni = Input((None, 128, 128, 3))
    scaled = [ResizeImages(int(128 * c))(ni) for c in scaling_factors]
    layers = [build_one_image_encoder(int(128 * c)).as_layer() for c in scaling_factors]
    no = [layer(im) for layer, im in zip(layers, scaled)]
    return tl.models.Model(inputs = ni, outputs = no)
