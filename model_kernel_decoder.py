from tensorlayer.layers import *
import tensorlayer as tl
import tensorflow as tf
from layer_split import Split

def build_kernel_decoder():
    ni = Input((None, 3200))
    nn = Reshape((-1, 5, 5, 128))(ni)
    nn = Conv2d(128, (5, 5), (1, 1), tf.nn.leaky_relu)(nn)
    nn = Conv2d(128, (5, 5), (1, 1))(nn)
    nn = Split(4, 3)(nn)
    return tl.models.Model(inputs = ni, outputs = nn)
