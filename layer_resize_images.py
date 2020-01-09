import tensorflow as tf
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers import *

# 为啥不用Upsampling2d和Downsampling2d，因为只关心放缩后多大，写着方便一些。
class ResizeImages(Layer):
    def __init__(self, sz):
        super(ResizeImages, self).__init__(name)
        self.sz = sz
        logging.info('ResizeImages %s' % self.name)
        self.build()
        self._built = True

    def build(self, inputs_shape = None):
        pass

    def __repr__(self):
        s = '{classname}(sz={sz})'.format(classname=self.__class__.__name__, **self.__dict__)
    
    def forward(self, inputs):
        return tf.image.resize_images(inputs, [self.sz, self.sz])
