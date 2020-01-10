import tensorflow as tf
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers import *

class CrossConv(Layer):
    def __init__(self, name = None):
        super(CrossConv, self).__init__(name)
        logging.info('CrossConv %s' % self.name)
        self.build()
        self._built = True

    def build(self, inputs_shape = None):
        pass

    def __repr__(self):
        return '{classname}()'.format(classname=self.__class__.__name__)
    
    def forward(self, pack):
        inputs, kernels = pack
        assert(inputs.shape[0] == kernels.shape[0] and
               inputs.shape[3] == kernels.shape[3])
        # 我们必须分离batch中的每个数据，这样才能对不同的图像使用不同的卷积核
        inputs = tf.unstack(inputs, axis = 0)
        kernels = tf.unstack(kernels, axis = 0)
        results = []
        for ni, ker in zip(inputs, kernels):
            # ni  ~ [imgsz, imgsz, channels] => [1, imgsz, imgsz, channels]
            # ker ~ [kersz, kersz, channels] => [kersz, kersz, channels, 1]
            # use tf.nn.depthwise_conv2d
            ni = tf.expand_dims(ni, 0)
            ker = tf.expand_dims(ker, 3)
            ret = tf.nn.depthwise_conv2d(ni, ker, [1, 1, 1, 1], 'SAME')
            results.append(ret)
        return tf.concat(results, 0)
