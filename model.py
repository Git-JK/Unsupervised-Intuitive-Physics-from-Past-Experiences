import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf

from model_image_encoder import build_image_encoder
from model_motion_encoder import build_motion_encoder
from model_kernel_decoder import build_kernel_decoder
from model_motion_decoder import build_motion_decoder
from layer_crossconv import CrossConv

def build():
    image_encoder = build_image_encoder().as_layer()
    motion_encoder = build_motion_encoder().as_layer()
    kernel_decoder = build_kernel_decoder().as_layer()
    motion_decoder = build_motion_decoder().as_layer()

    def build_eval():
        im = Input((None, 128, 128, 3))
        z = Input((None, 3200))
        features = image_encoder(im)
        kernels = kernel_decoder(z)
        features = [CrossConv()([feature, kernel]) for feature, kernel in zip(features, kernels)]
        output = motion_decoder(features)
        return tl.models.Model(inputs = [im, z], outputs = output)

    model_eval = build_eval()
    model_eval.eval()
    
    def build_train():
        im_before = Input((None, 128, 128, 3))
        im_after = Input((None, 128, 128, 3))
        im = Concat(3)([im_before, im_after])
        z0 = Input((None, 3200))
        mean, log_var = motion_encoder(im)
        # z <- mean + z0 * exp(log_var)
        z = Lambda(lambda pack: pack[1] + pack[0] * tf.exp(pack[2]))([z0, mean, log_var])
        output = model_eval.as_layer()([im_before, z])
        return tl.models.Model(inputs = [im_before, im_after, z0],
                               outputs = [mean, log_var, output])

    model_train = build_train()
    model_train.train()

    return model_eval, model_train
