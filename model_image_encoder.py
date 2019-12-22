import tensorlayer as tl
import tensorflow as tf
import config

class SingleImageEncoder(tl.models.Model):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        self.layers = []
        for layer_type, param in config.image_encoder_cnn:
            if layer_type == 'conv':
                self.layers.append(tl.layers.Conv2d(n_filter = param,
                                                    filter_size = (5, 5)))
            elif layer_type == 'maxpool':
                self.layers.append(tl.layers.PoolLayer(filter_size = (1, param, param, 1),
                                                       strides = (1, param, param, 1),
                                                       padding = 'SAME',
                                                       pool = tf.nn.max_pool))

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

class ImageEncoder(object):
    def __init__(self):
        self.encoders = []
        for scale in config.image_scaling:
            self.encoders.append(SingleImageEncoder())

    def forward(self, inputs):
        # inputs不是tensor，而是4个tensor的tuple
        outputs = [encoder.forward(oneinput)
                   for oneinput, encoder in zip(inputs, self.encoders)]
        return outputs
