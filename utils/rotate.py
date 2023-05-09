import os
import numpy as np
import albumentations as A
from PIL import Image


def rotate_im(img, deg, im_name, im_path):

    # Define the rotation range
    rotate_limit = (-deg, deg)

    # Create a rotation transform
    transform = A.Rotate(limit=rotate_limit, p=1)

    # Apply the transform to the image
    transformed_image = transform(image=np.array(image))["image"]

    # Save the transformed image
    Image.fromarray(transformed_image).save(im_path + im_name + "_rotated.png", 'PNG')

repo_path = os.environ.get('THESIS_PATH')
im_path =  repo_path + '/data/processed/crops3/class1/'
fnames = os.listdir(im_path)

for fname in fnames:
    image = Image.open(im_path + fname)
    new_fname = fname.split('.png')[0]
    rotate_im(image, 10, new_fname, im_path)
