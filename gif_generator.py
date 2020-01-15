import os
import imageio
from PIL import Image
import tensorlayer as tl
import tensorflow as tf
import numpy as np
from data import im_visualize_before as ims

origin_path = os.path.join('demo', 'origin.png')
tl.visualize.save_images(np.concatenate([ims] * 8, axis = 0), (8, ims.shape[0]),
                         origin_path)
im_origin = Image.open(origin_path)
print('GG: loaded origin')

for folder in os.listdir('demo'):
    folderp = os.path.join('demo', folder)
    if not os.path.isdir(folderp): continue
    for fname in os.listdir(folderp):
        if fname[-4:] == '.png' and fname[:-4].isnumeric():
            im = Image.open(os.path.join(folderp, fname))
            imgifp = os.path.join(folderp, fname[:-4] + '.gif')
            imageio.mimsave(imgifp, [im_origin, im], duration = 0.5)
            print('GG: written', imgifp)
