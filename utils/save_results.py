from PIL import Image, ImageDraw
import os

def crop_and_save_images(subdirectories, file_names, crop_area, output_dir):
    for model_dir, version_dir in subdirectories.items():
        version = '' if len(version_dir.split('/')) == 1 else version_dir.split('/')[-1]

        subdir_path = os.path.join(root_dir, model_dir, version_dir)
        if not os.path.isdir(subdir_path):
            continue

        for i,fname in enumerate(file_names):
            fpath, file_name = fname.split('/')
            file_path = os.path.join(subdir_path, fpath, file_name)
            if not os.path.isfile(file_path):
                continue

            image = Image.open(file_path)

            # Crop the image
            cropped_image = image.crop(crop_area[i])

            # Save the original image
            original_image_path = os.path.join(output_dir, f"original_{model_dir}{version}_{file_name}")
            image.save(original_image_path)

            # Save the cropped image
            cropped_image_path = os.path.join(output_dir, f"cropped_{model_dir}{version}_{file_name}")
            cropped_image.save(cropped_image_path)

            if model_dir == 'original_HR':
                # Create a bounding box on the original image
                draw = ImageDraw.Draw(image)
                draw.rectangle(crop_area[i], outline="red", width=3)

                # Save the image with the bounding box
                bbox_image_path = os.path.join(output_dir, f"bbox_{model_dir}{version}_{file_name}")
                image.save(bbox_image_path)

            print(f"Processed: {file_path}")

# Specify the root directory where your subdirectories are located
root_dir = "/work3/s164397/Thesis/Oblique2019/results"

# Specify the list of subdirectories to process
#subdirectories = {'EDSR':'1', 'ESRGAN':'test_images/3', 'SRGAN':'test_images/3', 'bicubic':'test_images', 'nearest_neighbor':'test_images'}
subdirectories = {'ESRGAN':'test_images/3', 'SRGAN':'test_images/3', 'bicubic':'test_images', 'nearest_neighbor':'test_images', 'original_HR':''}

# Specify the list of file names to process

# Specific crops in each test set
crops2 = ['11', '26', '72', '77', '82', '168', '195', '260', '281']
crops3 = ['2', '4', '11', '23', '29', '34', '41', '43', '49', '82', '337', '434']

fnames_2 = ['test_crops2_results/' + '2019_83_32_4_0024_00071126_crop_' + x + '.png' for x in crops2]
fnames_3 = ['test_crops3_results/' + '2019_81_08_2_0056_00001094_crop_' + x + '.png' for x in crops3]

file_names = fnames_2 + fnames_3

# Specify the crop area (left, upper, right, lower)
crop_areas2 = [(180, 50, 220, 90), (190, 260, 250, 320), (180, 180, 250, 250) , (220, 90, 285, 155), (110, 180, 170, 240), (50, 110, 120, 180), (280, 70, 340, 130), (100, 230, 200, 330), (10, 270, 80, 340)]
crop_areas3 = [(20, 290, 80, 350), (90, 280, 150, 340), (130, 300, 190, 360), (120, 60, 180, 120), (230, 220, 290, 280), (200, 240, 260, 300), (280, 280, 330, 330), (280, 300, 340, 360), (240, 210, 310, 280), (120, 210, 190, 280), (40, 240, 110, 310), (165, 150, 225, 210)]

crop_area = crop_areas2 + crop_areas3

# Specify the output directory to save the images
output_dir = "/work3/s164397/Thesis/Oblique2019/results/report"

crop_and_save_images(subdirectories, file_names, crop_area, output_dir)
