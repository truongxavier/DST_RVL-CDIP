from PIL import UnidentifiedImageError
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64




chemin_images = '/mnt/d/DVPT/DST/images/'
chemin_labels = '/mnt/d/DVPT/DST/labels/'
chemin_datasets = '/mnt/d/DVPT/DST/datasets/'
chemin_results = '/mnt/d/DVPT/DST/results/'

# Lire le fichier de labels en utilisant dask
df = pd.read_csv(chemin_results + 'processed_batch_0.csv')
plt.figure(figsize = (8,5))

def list_to_image(image_list, shape):
    return np.array(image_list).reshape(shape)

df['image'] = df.apply(lambda row: list_to_image(eval(row['image']), eval(row['shape'])), axis=1)

plt.imshow(df['image'].iloc[1], cmap='gray')
plt.show()

df.head()
df['label'].value_counts().head(15).plot(kind='bar')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.show()
plt.savfig('distribution_labels.png')

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
