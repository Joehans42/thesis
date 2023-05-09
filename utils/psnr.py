import numpy as np
from math import log10, sqrt
import cv2
from skimage.metrics import structural_similarity as ssim

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10((max_pixel / sqrt(mse)))
    return psnr

dir = 'results/100_sliced/'

gen_name = dir + 'valid_gen.png'
bic_name = dir + 'valid_hr_cubic.png'
hr_name = dir + 'valid_hr.png'

im_gen, im_bic, im_hr = cv2.imread(gen_name), cv2.imread(bic_name), cv2.imread(hr_name)

print(f'\nPSNR of Generated Image: {PSNR(im_gen, im_hr)}')
print(f'PSNR of Bicubic Image: {PSNR(im_bic, im_hr)}\n')

print(f'SSIM of Generated Image: {ssim(im_gen, im_hr, multichannel=True)}')
print(f'SSIM of Bicubic Image: {ssim(im_bic, im_hr, multichannel=True)}\n')

