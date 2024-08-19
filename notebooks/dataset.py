from PIL import UnidentifiedImageError
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


chemin_images = '/mnt/d/DVPT/DST/images/'
chemin_labels = '/mnt/d//DVPT/DST/labels/'
df = pd.read_csv(chemin_labels+'testmini.txt', sep=' ')
df.columns = ['image_chemin', 'label']
df['image_chemin'] = chemin_images + df['image_chemin']
def read_image(file_path):
    try:
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    except (FileNotFoundError, UnidentifiedImageError):
        return None

df['image'] = df['image_chemin'].apply(read_image)
df.head()

plt.figure(figsize = (8,5))

plt.imshow(df['image'].iloc[0], cmap='gray')
