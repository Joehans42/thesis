import os
import numpy as np
import pandas as pd
from PIL import Image
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim

def crop_and_save_images(subdirectories, file_names, output_dir):
    for model_dir, version_dir in subdirectories.items():
        version = '' if len(version_dir.split('/')) == 1 else version_dir.split('/')[-1]

        subdir_path = os.path.join(root_dir, model_dir, version_dir)
        
        if not os.path.isdir(subdir_path):
            continue

        psnrlist = []
        ssimlist = []
        fnames = []

        for i,fname in enumerate(file_names):
            fpath, file_name = fname.split('/')
            file_path = os.path.join(subdir_path, fpath, file_name)
            original_path = os.path.join(root_dir, 'original_HR', fpath, file_name)
            if not os.path.isfile(file_path):
                continue

            image = np.asarray(Image.open(file_path))
            orig_img = np.asarray(Image.open(original_path))

            psnrlist.append(round(PSNR(image, orig_img), 2))
            ssimlist.append(round(ssim(image, orig_img, multichannel=True, channel_axis=2, data_range=1), 2))
            fnames.append(file_name)
            
            print(f"Processed: {file_path}")

        csvname = f'{model_dir}{version}_PSNR_SSIM.csv'
        outpath = os.path.join(output_dir, csvname)

        df = pd.DataFrame(data={'PSNR':psnrlist, 'SSIM':ssimlist}, index=fnames)
        df.to_csv(outpath)
        
def PSNR(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10((max_pixel / sqrt(mse)))
    return psnr

# Specify the root directory where your subdirectories are located
root_dir = "/work3/s164397/Thesis/Oblique2019/results"

# Specify the list of subdirectories to process
#subdirectories = {'EDSR':'1', 'ESRGAN':'test_images/3', 'SRGAN':'test_images/3', 'bicubic':'test_images', 'nearest_neighbor':'test_images'}
subdirectories = {'ESRGAN':'test_images/3', 'SRGAN':'test_images/3', 'bicubic':'test_images', 'nearest_neighbor':'test_images'}

# Specify the list of file names to process

# Specific crops in each test set
crops2 = ['11', '26', '72', '77', '82', '168', '195', '260', '281']
crops3 = ['2', '4', '11', '23', '29', '34', '41', '43', '49', '82', '337', '434']

fnames_2 = ['test_crops2_results/' + '2019_83_32_4_0024_00071126_crop_' + x + '.png' for x in crops2]
fnames_3 = ['test_crops3_results/' + '2019_81_08_2_0056_00001094_crop_' + x + '.png' for x in crops3]

file_names = fnames_2 + fnames_3

output_dir = "/work3/s164397/Thesis/Oblique2019/results/report/psnr"

crop_and_save_images(subdirectories, file_names, output_dir)
