from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_image_size(img_name):
    try:
        with Image.open(img_name) as img:
            return img.size
    except UnidentifiedImageError:
        logging.info(f"Cannot identify image file {img_name}. Skipping.")
        return None

def get_max_image_size(csv_file, root_dir):
    labels_df = pd.read_csv(csv_file, sep=" ", header=None)
    max_width, max_height = 0, 0

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_image_size, os.path.join(root_dir, labels_df.iloc[idx, 0])): idx for idx in range(len(labels_df))}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating max image size"):
            size = future.result()
            if size:
                width, height = size
                if width > max_width:
                    max_width = width
                if height > max_height:
                    max_height = height

    return max_width, max_height

chemin_images = '/mnt/d/DVPT/DST/images/'
chemin_labels = '/mnt/d/DVPT/DST/labels/'
chemin_datasets = '/mnt/d/DVPT/DST/datasets/'
chemin_results = '/mnt/d/DVPT/DST/results/'

# Déterminer la taille maximale des images
max_width, max_height = get_max_image_size(chemin_labels + 'test.txt', chemin_images)
image_size = (max_width, max_height)
print(max_width, max_height)
print(image_size)
