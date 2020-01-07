
## 训练设置

cnt_epoch = 128
learning_rate = 0.001
kl_weight = 0.005  # https://github.com/yzwxx/vae-celebA/blob/master/train_vae.py#L99

## 网络设置

# 四种不同的scale
# 参数摘自论文
image_scaling = [2.0, 1.0, 0.5, 0.25]

# ImageEncoder中CNN的配置，这里写成了精简的语法
# 其中的参数摘自论文
image_encoder_cnn = [('conv', 64), ('maxpool', 2),
                     ('conv', 64),
                     ('conv', 64), ('maxpool', 2),
                     ('conv', 32)]
