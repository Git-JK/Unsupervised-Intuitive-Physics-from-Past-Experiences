from train_data import data
from model import model_train, model_test
import config

def enum_train_data():
    # to be implemented: to split data into batch
    raise NotImplementedError("enum_train_data not implemented")

def enum_test_data():
    # to be implemented: to get test data??
    raise NotImplementedError("enum_test_data not implemented")

def kl_loss(mean, logstdev):
    return tf.reduce_sum(-0.5 * tf.reduce_sum(1 + logstdev - tf.exp(logstdev) - mean ** 2))

optimizer = tf.optimizers.Adam(learning_rate = config.learning_rate)

for epoch in range(config.cnt_epoch):
    print('epoch', epoch)
    for step, (data_before, data_after) in enumerate(enum_train_data()):
        model_train.train()
        with tf.GradientTape(persistent = True) as tape:
            outputs, mean, logstdev = model_train(data_before, data_after)
            loss = kl_loss(mean, logstdev) + l2_loss(outputs, data_before)
            print(step, loss)
        grad = tape.gradient(loss, model_train.trainable_weights)
        optimizer.apply_gradients(zip(grad, model_train.trainable_weights))
