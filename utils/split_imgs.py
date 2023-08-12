import os
from PIL import Image

def split_image(image_path, output_folder):
    # Open the image
    image = Image.open(image_path)

    # Calculate the width and height of each part
    width, height = image.size
    part_width = width // 4
    part_height = height // 4

    # Split the image into 16 parts
    parts = []
    for i in range(4):
        for j in range(4):
            left = j * part_width
            upper = i * part_height
            right = left + part_width
            lower = upper + part_height
            part = image.crop((left, upper, right, lower))
            parts.append(part)

    # Save the parts as new images
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    for i, part in enumerate(parts):
        part_path = os.path.join(output_folder, f"{image_name}_part{i+1}.png")
        part.save(part_path)
        print(f"Saved {part_path}")

#repo_path = os.environ.get('THESIS_PATH')

# Set the input and output folder paths
input_folder = "../data/processed/test_crops"
output_folder = "../data/processed/test_crops_x4"

# Get all the image files in the input folder
image_files = [f for f in os.listdir(input_folder)]

# Split each image and save the parts
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    split_image(image_path, output_folder)
