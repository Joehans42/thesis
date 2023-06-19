import albumentations as A
from PIL import Image
import os
import numpy as np

HIGH_RES = 384
LOW_RES = HIGH_RES // 4

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        #A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ]
)

im_path = '../data/processed/test_crops2/'
save_path = '../src/models/EDSR/test_images/'

file_list = [f for f in os.listdir(im_path)]


for fname in file_list:
    img = np.array(Image.open(im_path + fname))
    lr = lowres_transform(image=img)["image"]

    lr_im = Image.fromarray(lr)
    lr_im.save(os.path.join(save_path, fname), 'PNG')
