from tensorlayer.layers import *
import tensorlayer as tl
import tensorflow as tf
from layer_split import Split

def build_motion_encoder():
    ni = Input((None, 128, 128, 6))
    nn = Conv2d(96, (5, 5), (2, 2), tf.nn.relu)(ni)
    nn = Conv2d(96, (5, 5), (1, 1), tf.nn.relu)(nn)
    nn = Conv2d(128, (5, 5), (2, 2), tf.nn.relu, 'VALID')(nn)
    nn = Conv2d(128, (5, 5), (2, 2), tf.nn.relu, 'VALID')(nn)
    nn = Conv2d(256, (5, 5), (1, 1), tf.nn.relu, 'VALID')(nn)
    nn = Conv2d(256, (5, 5), (1, 1), None, 'VALID')(nn)
    nn = Reshape((-1, 6400))(nn)
    mean, log_var = Split(2, 1)(nn)
    return tl.models.Model(inputs = ni, outputs = [mean, log_var])
