import numpy as np
from math import log10, sqrt
import cv2
from skimage.metrics import structural_similarity as ssim
import sys

def PSNR(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10((max_pixel / sqrt(mse)))
    return psnr

if len(sys.argv) == 3:
    im1 = cv2.imread(sys.argv[1])
    im2 = cv2.imread(sys.argv[2])

    print(f'\nPSNR: {PSNR(im1, im2)}')
    print(f'SSIM: {ssim(im1, im2, multichannel=True, channel_axis=2, data_range=1)}')
else:
    print('give 2 filenames dummy')
