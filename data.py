from PIL import Image
import numpy as np
import os
import config

def read_from_path(path):
    #path = "./exercise_dataset/release_data_set/images"
    filenames = os.listdir(path)
    im_before = None
    im_after = None
    count = 0
    for filename in filenames:
        if filename[len(filename)-5] == '1': 
            fp = open(os.path.join(path,filename),'rb')
            if im_before is None:
                im_before = [np.array(Image.open(fp))]
            else:
                im_before.append(np.array(Image.open(fp)))
                
            fp.close()
        else:
            if filename[len(filename)-5] == '2': 
                fp = open(os.path.join(path,filename),'rb')
                if im_after is None:
                    im_after = [np.array(Image.open(fp))]
                else:
                    im_after.append(np.array(Image.open(fp)))
                fp.close()       
    return np.array(im_before),np.array(im_after)
    
data_before, data_after = read_from_path(config.data_dir)
print(data_before.shape)
print(data_after.shape)
