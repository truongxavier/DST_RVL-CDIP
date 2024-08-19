from PIL import UnidentifiedImageError
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pytesseract


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
df['shape'] = df['image'].apply(lambda x: x.shape if x is not None else None)
df.head()

plt.figure(figsize = (8,5))

plt.imshow(df['image'].iloc[1], cmap='gray')

df.head()
df['label'].value_counts().head(15).plot(kind='bar')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.show()

df['shape'].value_counts().plot(kind='bar')
plt.xlabel('Shape')
plt.ylabel('Count')
plt.title('Distribution of Shapes')
plt.show()

df['image'].apply(lambda x: plt.hist(x.flatten(), bins=256, color='gray', alpha=0.7))
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Pixel Value Distribution')
plt.show()

def ocr_image(image):
    if image is not None:
        return pytesseract.image_to_string(image)
    else:
        return None

df['ocr_text'] = df['image'].apply(ocr_image)
df.head()
