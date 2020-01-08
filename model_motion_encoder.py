import tensorflow as tf
from tensorlayer.layers import *
import tensorlayer as tl
import numpy as np
import config

def build_motion_encoder(input_size):
    ni = Input((None, input_size, input_size, 6))
    nn = ni
    for layer_type, param in config.motion_encoder_cnn:
        if layer_type == 'conv':
            nn = Conv2d(n_filter = param, filter_size = (5, 5))(nn)
        else:
            nn = PoolLayer(filter_size = (1, param, param, 1),
                           strides = (1, param, param, 1),
                           padding = 'SAME',
                           pool = tf.nn.max_pool)(nn)
    output_size = nn.shape[1:].num_elements()
    mean, log_var = tf.split(tf.reshape(nn, (-1, output_size)),
                             [output_size // 2], 1)
    return tl.models.Model(inputs = ni, outputs = [mean, log_var])
