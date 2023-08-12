import albumentations as A
from PIL import Image
import os
import numpy as np

HIGH_RES = 500
LOW_RES = HIGH_RES // 4

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        #A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ]
)

im_path = '../data/processed/edsr/Custom/Oblique_valid_HR/'
save_path = '../data/processed/edsr/Custom/Oblique_valid_LR_bicubic/X4/'

file_list = [f for f in os.listdir(im_path)]


for fname in file_list:
    img = np.array(Image.open(im_path + fname))
    lr = lowres_transform(image=img)["image"]

    new_fname = fname.split('.')[0] + 'x4' + '.png'

    lr_im = Image.fromarray(lr)
    lr_im.save(os.path.join(save_path, new_fname), 'PNG')
