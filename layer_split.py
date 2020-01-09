import tensorflow as tf
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers import *

class Split(Layer):
    def __init__(self, num_or_size_splits, axis = 0):
        super(Split, self).__init__(name)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis
        logging.info('Split %s' % self.name)
        self.build()
        self._built = True

    def build(self, inputs_shape = None):
        pass

    def __repr__(self):
        s = '{classname}(num_or_size_splits={number_or_size_splits}, axis={axis}, name={name})'.format(classname=self.__class__.__name__, **self.__dict__)
    
    def forward(self, inputs):
        return tf.split(inputs, self.num_or_size_splits, self.axis)
