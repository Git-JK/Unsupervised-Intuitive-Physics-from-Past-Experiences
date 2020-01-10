import tensorlayer as tl
import numpy as np
import os
import config
from tqdm import tqdm

def read_imlist_from_file(fpath):
    with open(fpath) as f:
        return f.read().split()

def read_imlist(path):
    ret = os.listdir(path)
    ret = [s[:9] for s in ret if s[-8:] == '_im1.png']
    return ret

def preprocess(x):
    return x / 127.5 - 1.

def revert_preprocess(x):
    return np.clip((x + 1.) * 127.5, 0., 255.)

def read_im(path, filelist):
    im_before = tl.visualize.read_images([f + '_im1.png' for f in filelist], path)
    im_before = np.stack(im_before, 0).astype(np.float32)
    im_before = preprocess(im_before)
    im_after = tl.visualize.read_images([f + '_im2.png' for f in filelist], path)
    im_after = np.stack(im_after, 0).astype(np.float32)
    im_after = preprocess(im_after)
    return im_before, im_after

imlist_train = read_imlist_from_file(config.data_train_list)
imlist_test = read_imlist_from_file(config.data_test_list)
imlist_visualize = read_imlist_from_file(config.data_visualize_list)

len_test = len(imlist_test)

im_visualize_before, im_visualize_after = read_im(config.data_dir, imlist_visualize)

def enum(imlist_full):
    for imlist in tl.iterate.minibatches(imlist_full, imlist_full, config.batch_size,
                                         allow_dynamic_batch_size = True, shuffle = True):
        yield read_im(config.data_dir, imlist[0])

def enum_with_progress(imlist_full):
    with tqdm(total = len(imlist_full)) as pbar:
        for x, y in enum(imlist_full):
            yield x, y
            pbar.update(x.shape[0])

def enum_train_with_progress():
    return enum_with_progress(imlist_train)

def enum_test_with_progress():
    return enum_with_progress(imlist_test)
