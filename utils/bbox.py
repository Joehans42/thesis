from PIL import Image, ImageDraw
import os

def save_image_with_bounding_boxes(fname, coordinates_list, output_path):
    image = Image.open(os.path.join(image_path, fname))
    hr = Image.open(hr_im)
    draw = ImageDraw.Draw(image)

    for i, coordinates in enumerate(coordinates_list):
        x1, y1, x2, y2 = coordinates
        
        hr_crop = hr.crop((x1, y1, x2, y2))
        hr_crop.save(os.path.join(output_path, f"{fname}_part{i+1}HR.png"), 'PNG')
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image.save(os.path.join(output_path, f"{fname}_part{i+1}.png"), 'PNG')

        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    image.save(os.path.join(output_path, f"{fname}_bounding_boxes.png"), 'PNG')

# Example usage
image_path = "results"
coordinates_list = [[(220, 120, 260, 160), (210, 230, 250, 270)], [(260, 210, 300, 250), (330, 120, 370, 160)], [(190, 130, 230, 170), (190, 250, 230, 290)], [(100, 30, 140, 70), (30, 290, 70, 330)], [(190, 70, 230, 110), (70, 110, 110, 150)]]
output_path = "results/bbox"
file_list = [f for f in os.listdir(image_path) if not os.path.isdir(os.path.join(image_path, f))]
hr_im = "../data/processed/test_crops/crop_11_23.jpg"
#print(file_list)
for i, fname in enumerate(file_list):
    save_image_with_bounding_boxes(fname, coordinates_list[i], output_path)
