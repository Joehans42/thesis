from PIL import Image
import os

# Paths
folder_path = "../../../data/processed/test_crops3"
output_folder = "./test_images/test_crops3_results"

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Open the image
    image_path = os.path.join(folder_path, filename)
    image = Image.open(image_path)

    # Perform nearest neighbor interpolation, downscaling then back up
    width, height = image.size
    interpolated_image = image.resize((width//4, height//4), Image.BICUBIC)
    interpolated_image = interpolated_image.resize((width, height), Image.NEAREST)

    # Save the interpolated image
    output_path = os.path.join(output_folder, filename)
    interpolated_image.save(output_path)

    print(f"Interpolated image saved: {output_path}")
