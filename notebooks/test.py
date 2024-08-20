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

def deserialize_image(image_str):
    image_data = base64.b64decode(image_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

image_data = df['image'].iloc[1]

if isinstance(image_data, str):
    try:
        image_data = deserialize_image(image_data)
    except ValueError as e:
        print(f"Erreur de conversion de l'image : {e}")
        print(f"Chaîne de caractères de l'image : {image_data}")

plt.imshow(image_data, cmap='gray')
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
