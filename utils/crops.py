import os
import random
import numpy as np
import pandas as pd
from PIL import Image

def extract_and_save_random_crops(img, patch_size, crop_size, save_path, im_num):
    # Convert the image to a numpy array
    img_array = np.array(img)

    # Get the dimensions of the image
    height, width, _ = img_array.shape

    # Calculate the number of patches we can extract
    num_patches_h = int(np.floor(height / patch_size))
    num_patches_w = int(np.floor(width / patch_size))
    num_patches = num_patches_h * num_patches_w

    # Loop over the patches and extract random crops
    crops = []
    for i in range(num_patches):
        # Calculate the patch coordinates
        patch_x = (i % num_patches_w) * patch_size
        patch_y = (i // num_patches_w) * patch_size

        # Extract a random crop from the patch
        x = random.randint(patch_x, patch_x + patch_size - crop_size)
        y = random.randint(patch_y, patch_y + patch_size - crop_size)
        crop = img.crop((x, y, x+crop_size, y+crop_size))
        crops.append(np.array(crop))

    # Save the crops as images
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, crop in enumerate(crops):
        crop_img = Image.fromarray(crop)
        crop_img.save(os.path.join(save_path, f"{im_num}_crop_{i}.jpg"))


num_ims = 100
classes = pd.read_csv('classification.csv')
indexes = classes[classes.classification==1].sample(n=num_ims).index
file_list = classes.iloc[indexes].filename
im_path = '../data/raw/class1/'
save_path = '../data/processed/crops2/'

for i, fname in enumerate(file_list):
    img = Image.open(im_path + fname)
    extract_and_save_random_crops(img, 500, 384, save_path, fname)


