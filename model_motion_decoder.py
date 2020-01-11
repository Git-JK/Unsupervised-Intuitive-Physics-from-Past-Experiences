import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf

# 四个Input的大小
input_sizes = [64, 32, 16, 8]

def build_motion_decoder():
    ni = [Input((None, sz, sz, 32)) for sz in input_sizes]
    scaled = [DeConv2d(32, (9, 9), (128 // sz, 128 // sz))(ni)
              for x, sz in zip(ni, input_sizes)]
    x = Concat(3)(scaled)
    nn = BatchNorm2d()(x)
    nn = Conv2d(128, (5, 5), (1, 1), tf.nn.leaky_relu)(nn)
    nn = BatchNorm2d()(nn)
    nn = Conv2d(3, (5, 5), (1, 1))(nn)
    return tl.models.Model(inputs = ni, outputs = nn)
