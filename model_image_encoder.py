import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf
import config

def build_one(input_size):
    ni = Input((None, input_size, input_size, 3))
    nn = ni
    for layer_type, param in config.image_encoder_cnn:
        if layer_type == 'conv':
            nn = Conv2d(n_filter = param, filter_size = (5, 5))(nn)
        else:
            nn = PoolLayer(filter_size = (1, param, param, 1),
                           strides = (1, param, param, 1),
                           padding = 'SAME',
                           pool = tf.nn.max_pool)(nn)
    no = nn
    return tl.models.Model(inputs = ni, outputs = no)

def build_image_encoder(image_size):
    models = [build_one(int(image_size * scale)) for scale in config.image_scaling]
    def forward(inputs):
        return [model(ni) for ni in inputs]
    return forward
