import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf
from layer_resize_images import ResizeImages

def build_motion_decoder():
    ni = [Input((None, sz, sz, 32)) for sz in [64, 32, 16, 8]]
    scaled = [ResizeImages(128)(x) for x in ni]
    x = Concat(3)(scaled)
    nn = Conv2d(128, (9, 9), (1, 1), tf.nn.relu)(x)
    nn = BatchNorm2d()(nn)
    nn = Conv2d(128, (1, 1), (1, 1), tf.nn.relu)(nn)
    nn = BatchNorm2d()(nn)
    nn = Conv2d(3, (1, 1), (1, 1))(nn)
    return tl.models.Model(inputs = ni, outputs = nn)
