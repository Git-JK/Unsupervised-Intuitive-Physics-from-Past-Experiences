import tensorflow as tf
from tensorlayer.layers import *
import tensorlayer as tl
import numpy as np
import config

def gaussian_sampler(mean, log_var):
    outputs = []
    length = len(mean)
    for i in range(length):
        outputs.append(np.random.normal(mean[i], exp(log_var[i])))
    return outputs

def conv_cross2d(inputs, weights):
    outputs = []
    for input, weight in zip(inputs, weights):
        output = Conv2d(
            n_filter = weight[0] * weight[1],
            filter_size = (weight[2], weight[3]),
            padding = 'SAME'
        )(input)
        outputs.append(output)
    outputs = tf.concat(outputs, 0)
    return outputs

class image_encoder:
    def __init__(self, image_size):
        self.models = [build_one(int(image_size * scale)) for scale in config.image_scaling]
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

    def forward(self, inputs):
        return [models(ni) for ni in inputs]

class motion_encoder:

    def __init__(self, image_size):
        self.models = [build_one(int(image_size * scale))] for scale in config.image_scaling

    def build_one(input_size):
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
        no = nn
        return tl.models.Model(inputs = ni, outputs = no)

    def forward(self, inputs):
        inputs = tf.concat(inputs, 1)
        outputs = self.models(inputs)
        size = tf.shape(tf.constant(inputs))
        outputs = tf.reshape(size[0], [-1])
        mean, log_var = tf.split(outputs, [size[1] // 2], 1)
        return mean, log_var

class kernel_decoder:
    def __init__(self, image_size):
        self.num_scales = config.kernel_decoder[0]
        self.in_channels = config.kernel_decoder[1]
        self.out_channels = config.kernel_decoder[2]
        self.kernel_size = config.kernel_decoder[3]
        self.num_groups = config.kernel_decoder[4]
        self.num_layers = config.kernel_decoder[5]
        self.kernel_sizes = config.kernel_decoder[6]
        self.num_channels = self.num_scales * self.out_channels * (self.in_channels // self.num_groups)
        self.layers = tf.keras.models.Sequential()
        self.layers.add(build_deconv2d(int(image_size)))
        self.layers.add(build_conv2d(int(image_size)))
        self.layers.add(tl.layers.BatchNorm2d(self.num_channels))

    def build_deconv2d(input_size): 
        ni = Input((None, input_size, input_size, 2))
        nn = ni
        for layer_type, param in config.kernel_decoder_decnn:
            if layer_type == 'deconv':
                 nn = DeConv2d(n_filter = param, filter_size = (5, 5))(nn)
            else:
                nn = PoolLayer(filter_size  = (1, param, param, 1)),
                               strides = (1, param, param, 1),
                               padding = 'SAME',
                               pool = tf.nn.max_pool)(nn)
        no = nn
        return tl.models.Model(inputs = ni, outputs = no)

    def build_conv2d(input_size):
        ni = Input((None, input_size, input_size, 3))
        nn = ni
        for layer_type, param in config.kernel_decoder_cnn:
            if layer_type == 'conv':
                 nn = Conv2d(n_filter = param, filter_size = (5, 5))(nn)
            else:
                nn = PoolLayer(filter_size  = (1, param, param, 1)),
                               strides = (1, param, param, 1),
                               padding = 'SAME',
                               pool = tf.nn.max_pool)(nn)
        no = nn
        return tl.models.Model(inputs = ni, outputs = no)

    def forward(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.num_channels, 1, 1])
        outputs = self.layers(inputs)
        outputs = tf.reshape(inputs, [-1, self.num_scales, self.out_channels, self.in_channels // self.num_groups, self.kernel_size, self.kernel_size])
        return outputs
    
    
class motion_decoder:
    def __init__(self, image_size):
        self.scales = config.image_scaling
        self.models = []
        self.models.append([buid_one(int(image_size * scale)) for scale in config.image_scaling])
    def build_one(input_size):
        ni = Input((None, input_size, input_size, 3))
        nn = ni
        nn = Conv2d(n_filter = 128, filter_size = (9, 9))(nn)
        no = nn
        return tl.models.Model(inputs = ni, outputs = no)
    def forward(self, inputs):
        for k, input in enumerate(inputs)
            scale_factor = int(self.scales[-1] / self.scales[k])
            if scale_factor != 1:
                inputs[k] = tf.compat.v1.image.resize_nearest_neighbor(input, scale_factor)

            inputs = tf.concat(inputs, 1)
            outputs = self.models(inputs)
            return outputs

class VDNet:
    def __init__(self):
        self.scales = config.image_scaling
        self.image_encoder = image_encoder()
        self.motion_encoder = motion_encoder()
        self.kernel_decoder = kernel_decoder()
        self.motion_decoder = motion_decoder()

    def forward(self, inputs, mean = None, log_var = None, z = None, returns = None)
        if isinstance(inputs, list) and len(inputs) == 2:
            i_inputs, m_inputs = inputs
        else:
            i_inputs, m_inputs = inputs, None
        
        #image encoder
        features = self.image_encoder.forward(i_inputs)

        #motion encoder
        if mean is None and log_var is None and z is None:
            mean, log_var = self.motion_encoder.forward(m_inputs)

        #guassian sampler
        if z is None:
            z = gaussian_sampler(mean, log_var)
        
        #kernel decoder

        kernels = self.kernel_decoder.forward(z)

        #cross convolution
        for i, feature in enumerate(features):
            kernel = kernels[: i, ...]
            features[i] = conv_cross2d(feature, kernel)
        
        #motion decoder
        outputs = self.motion_decoder.forward(features)

        #returns
        if returns is Not None:
            if not isinstance(returns, (list, tuple)):
                returns = [returns]
            for i, k in enumerate(returns):
                returns[i] = locals()[k]
            return outputs, returns[0] if len(returns) == 1 else returns
