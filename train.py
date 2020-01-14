from data import (enum_train_with_progress,
                  revert_preprocess,
                  im_visualize_before, im_visualize_after,
                  enum_test_with_progress,
                  len_train, len_test)
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

def read_progress():
    last_progress = -1
    for f in os.listdir(config.save_snapshot_to):
        if f[:12] == 'model_train_' and f[-3] == '.h5':
            last_progress = max(last_progress, int(f[12:-3]))
    if last_progress == -1:
        print('No snapshot found')
        return 0
    else:
        inp = input('Found max snapshot id: ' + str(last_progress) + ', Enter where to start (0 ~ inf): ')
        return int(inp)

last_progress = read_progress()
if last_progress > 0:
    model_train.load_weights(os.path.join(config.save_snapshot_to, 'model_train_' + str(epoch + 1) + '.h5'))

writer = tf.summary.create_file_writer(config.save_logs_to)

for epoch in range(last_progress, config.cnt_epoch):
    print('Epoch %d/%d' % (epoch, config.cnt_epoch))
    tf.summary.experimental.set_step(epoch)

    model_train.train()
    loss, loss_kl, loss_l2 = 0, 0, 0
    for data_before, data_after in enum_train_with_progress():
        z0 = np.random.normal(size = data_before.shape[0] * 3200).astype(np.float32).reshape((-1, 3200))
        with tf.GradientTape() as tape:
            mean, logstdev, outputs = model_train([data_before, data_after, z0])
            loss_kl_t = kl_loss(mean, logstdev)
            loss_l2_t = l2_loss(outputs, data_after - data_before)
            loss_t = config.kl_weight * loss_kl_t + loss_l2_t
        grad = tape.gradient(loss_t, model_train.trainable_weights)
        optimizer.apply_gradients(zip(grad, model_train.trainable_weights))

        loss_kl += loss_kl_t * data_before.shape[0] / len_train
        loss_l2 += loss_l2_t * data_before.shape[0] / len_train
        loss += loss_t * data_before.shape[0] / len_train
        
    with writer.as_default():
        tf.summary.scalar('train_loss_kl', loss_kl)
        tf.summary.scalar('train_loss_l2', loss_l2)
        tf.summary.scalar('train_loss', loss)
        writer.flush()

    model_train.eval()
    loss, loss_kl, loss_l2 = 0, 0, 0
    for data_before, data_after in enum_test_with_progress():
        z0 = np.random.normal(size = data_before.shape[0] * 3200).astype(np.float32).reshape((-1, 3200))
        mean, logstdev, outputs = model_train([data_before, data_after, z0])
        loss_kl_t = kl_loss(mean, logstdev)
        loss_l2_t = l2_loss(outputs, data_after - data_before)
        loss_t = config.kl_weight * loss_kl_t + loss_l2_t
        loss_kl += loss_kl_t * data_before.shape[0] / len_test
        loss_l2 += loss_l2_t * data_before.shape[0] / len_test
        loss += loss_t * data_before.shape[0] / len_test
    with writer.as_default():
        tf.summary.scalar('test_loss_kl', loss_kl)
        tf.summary.scalar('test_loss_l2', loss_l2)
        tf.summary.scalar('test_loss', loss)
        writer.flush()
    
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
        im_path = os.path.join(config.save_visualization_to, 'test_' + str(epoch + 1) + '.png')
        tl.visualize.save_images(images, (8, im_visualize_before.shape[0]), im_path)
        with writer.as_default():
            tf.summary.image('visualize', np.expand_dims(tl.visualize.read_image(im_path), axis = 0))
            writer.flush()
