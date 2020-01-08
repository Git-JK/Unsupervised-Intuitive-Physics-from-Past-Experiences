import tensorflow as tf
import tensorlayer as tl
import config

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
     
