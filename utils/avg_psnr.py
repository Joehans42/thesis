import os
import cv2
import numpy as np

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_average_psnr(directory1, directory2):
    images1 = os.listdir(directory1)
    images2 = os.listdir(directory2)

    total_psnr = 0.0
    num_images = 0

    for image_name in images1:
        if image_name in images2:
            image_path1 = os.path.join(directory1, image_name)
            image_path2 = os.path.join(directory2, image_name)

            image1 = cv2.imread(image_path1)
            image2 = cv2.imread(image_path2)

            psnr = calculate_psnr(image1, image2)
            total_psnr += psnr
            num_images += 1

    if num_images > 0:
        average_psnr = total_psnr / num_images
        return average_psnr
    else:
        return 0.0

# Specify the directories containing the images
root_dir = "/work3/s164397/Thesis/Oblique2019/results"
model = 'ESRGAN'
version = '4'

directory1 = os.path.join(root_dir, model, 'test_images', version)
directory2 = os.path.join(root_dir, 'original_HR')

testset1 = 'test_crops2_results'
testset2 = 'test_crops3_results'

average_psnr1 = calculate_average_psnr(os.path.join(directory1, testset1), os.path.join(directory2, testset1))
average_psnr2 = calculate_average_psnr(os.path.join(directory1, testset2), os.path.join(directory2, testset2))

print(f"Average PSNR of {model} v{version} on {testset1}: {round(average_psnr1, 2)}")
print(f"Average PSNR of {model} v{version} on {testset2}: {round(average_psnr2, 2)}")

total_avg = (average_psnr1 * 300 + average_psnr2 * 638)/938

print(f"Total average PSNR of {model} v{version}: {round(total_avg, 2)}")