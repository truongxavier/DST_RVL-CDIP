import dask.dataframe as dd
import dask.array as da
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pytesseract

# Spécifiez le chemin de Tesseract si nécessaire
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Remplacez par le chemin correct si nécessaire

chemin_images = '/mnt/d/DVPT/DST/images/'
chemin_labels = '/mnt/d/DVPT/DST/labels/'

# Lire le fichier de labels en utilisant dask
df = dd.read_csv(chemin_labels + 'test.txt', sep=' ', names=['image_chemin', 'label'])
df['image_chemin'] = chemin_images + df['image_chemin']

def read_image(file_path):
    try:
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    except (FileNotFoundError, UnidentifiedImageError):
        return None

def process_batch(batch):
    batch['image'] = batch['image_chemin'].apply(read_image, meta=('image_chemin', 'object'))
    batch['shape'] = batch['image'].apply(lambda x: x.shape if x is not None else None, meta=('image', 'object'))
    return batch

# Traiter les données par lots
batch_size = 1000  # Ajustez la taille du lot selon votre mémoire disponible
df = df.map_partitions(process_batch, meta={'image_chemin': 'object', 'label': 'int64', 'image': 'object', 'shape': 'object'})

# Convertir en DataFrame pandas pour les opérations de visualisation
df = df.compute()

# Visualisation
plt.figure(figsize=(8, 5))
plt.imshow(df['image'].iloc[1], cmap='gray')
plt.show()

df['label'].value_counts().head(15).plot(kind='bar')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.savefig('distribution_labels.png')
plt.show()

df['shape'].value_counts().plot(kind='bar')
plt.xlabel('Shape')
plt.ylabel('Count')
plt.title('Distribution of Shapes')
plt.show()

df['image'].apply(lambda x: plt.hist(x.flatten(), bins=256, color='gray', alpha=0.7) if x is not None else None)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Pixel Value Distribution')
plt.show()

# def ocr_image(image):
#     if image is not None:
#         return pytesseract.image_to_string(image)
#     else:
#         return None

# df['ocr_text'] = df['image'].apply(ocr_image)
# print(df.head())
