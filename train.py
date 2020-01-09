from data import enum_train, revert_preprocess, im_visualize_before, im_visualize_after
from model import build
import config
import os
import numpy as np
import tensorlayer as tl
import tensorflow as tf

model_eval, model_train = build()

optimizer = tf.optimizers.Adam(learning_rate = config.learning_rate)

def l2_loss(a, b):
    return tf.reduce_mean(tf.square(a - b))

def kl_loss(mean, logstdev):
    return -0.5 * tf.reduce_mean(1 + logstdev - tf.exp(logstdev) - mean ** 2)

tl.files.exists_or_mkdir(config.save_snapshot_to)
tl.files.exists_or_mkdir(config.save_visualization_to)

for epoch in range(config.cnt_epoch):
    print('epoch', epoch)

    for step, (data_before, data_after) in enumerate(enum_train()):
        print('step', step)
        z0 = np.random.normal(size = data_before.shape[0] * 3200).astype(np.float32).reshape((-1, 3200))
        with tf.GradientTape(persistent = True) as tape:
            mean, logstdev, outputs = model_train([data_before, data_after, z0])
            loss = config.kl_weight * kl_loss(mean, logstdev) + l2_loss(outputs, data_after - data_before)
            print(step, loss)
        grad = tape.gradient(loss, model_train.trainable_weights)
        optimizer.apply_gradients(zip(grad, model_train.trainable_weights))
    
    if (epoch + 1) % config.log_every == 0:
        model_train.save_weights(os.path.join(config.save_snapshot_to, 'model_train_' + str(epoch + 1) + '.h5'))
        # assuming test_data is just several images (~10)
        afters = []
        for i in range(6):
            z0 = np.random.normal(size = im_visualize_before.shape[0] * 3200).astype(np.float32).reshape((-1, 3200))
            visualize_after = model_eval([im_visualize_before, z0])
            afters.append(visualize_after + im_visualize_before)
        images = tf.concat([im_visualize_before] + afters + [im_visualize_after], axis = 0)
        images = revert_preprocess(images)
        tl.visualize.save_images(images, (8, im_visualize_before.shape[0]),
                                 os.path.join(config.save_visualization_to, 'test_' + str(epoch + 1) + '.png'))
        
