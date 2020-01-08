
## 数据目录设置

data_dir = 'data/exercise'

## 训练设置

cnt_epoch = 128
batch_size = 32
learning_rate = 0.001
kl_weight = 0.005  # https://github.com/yzwxx/vae-celebA/blob/master/train_vae.py#L99

log_every = 16
save_snapshot_to = 'result/models'
save_visualization_to = 'result/visualizations'

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

motion_encoder_cnn = [('conv', 96), ('maxpool', 4),
                      ('conv', 96), ('maxpool', 2),
                      ('conv', 128), ('maxpool', 2),  
                      ('conv', 128), ('maxpool', 2),  
                      ('conv', 256), ('maxpool', 2),
                      ('conv', 256), ('maxpool', 2)]

kernel_decoder = [4, 32, 32, 32, 5, 3, 5]
kernel_decoder_spec = {
    'num_scales': 4,
    'in_channels': 32,
    'out_channels': 32,
    'kernel_size': 32,
    'num_groups': 5,
    'num_layers': 3,
    'kernel_sizes': 5
}

kernel_decoder_decnn = [('deconv', 128), ('maxpool', 5),
                        ('deconv', 128), ('maxpool', 5)]

kernel_decoder_cnn = [('conv', 128), ('conv', 128), ('conv', 128)]

motion_decoder_cnn = [('conv', 128), ('maxpool', 9),
                      ('conv', 128),
                      ('conv', 128)]
