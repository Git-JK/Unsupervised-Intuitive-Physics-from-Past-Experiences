from train_data import train_data, test_data, revert_preprocess
from model import model_train, model_test
import config
import os

def enum_train_data(batch_size):
    # to be implemented: to split data into batch
    raise NotImplementedError("enum_train_data not implemented")

def l2_loss(a, b):
    return tf.reduce_sum(tf.square(a - b))

def kl_loss(mean, logstdev):
    return tf.reduce_sum(-0.5 * tf.reduce_sum(1 + logstdev - tf.exp(logstdev) - mean ** 2))

optimizer = tf.optimizers.Adam(learning_rate = config.learning_rate)

tl.files.exists_or_mkdir(config.save_snapshot_to)
tl.files.exists_or_mkdir(config.save_visualization_to)

for epoch in range(config.cnt_epoch):
    print('epoch', epoch)
    
    for step, (data_before, data_after) in enumerate(enum_train_data(config.batch_size)):
        model_train.train()
        with tf.GradientTape(persistent = True) as tape:
            outputs, mean, logstdev = model_train(data_before, data_after)
            loss = config.kl_weight * kl_loss(mean, logstdev) + l2_loss(outputs, data_after - data_before)
            print(step, loss)
        grad = tape.gradient(loss, model_train.trainable_weights)
        optimizer.apply_gradients(zip(grad, model_train.trainable_weights))
        
    if (epoch + 1) % config.log_every == 0:
        model_train.save_weights(os.path.join(save_snapshot_to, 'model_train_' + str(epoch + 1) + '.h5'))
        model_test.test()
        # assuming test_data is just several images (~10)
        test_data_after = model_test(test_data)
        images = tf.concat([test_data, test_data_after + test_data], 0)
        images = revert_preprocess(images)  # 逆向预处理（像素级）
        tl.visualize.save_images(images, (2, test_data.shape[0]),
                                 os.path.join(save_visualization_to, 'test_' + str(epoch + 1) + '.png'))
