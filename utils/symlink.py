import os
import glob
import pandas as pd

# Set the paths to the directories and files
source_dir = "/work3/s164397/Thesis/Oblique2019/images"
link_dir = "../data/raw/class1"
classes = pd.read_csv('classification.csv')

file_list = classes[classes.classification==1].filename

# Create symbolic links for each file in the list
for file in file_list:
    source_path = os.path.join(source_dir, file)
    link_path = os.path.join(link_dir, file)
    os.symlink(source_path, link_path)

