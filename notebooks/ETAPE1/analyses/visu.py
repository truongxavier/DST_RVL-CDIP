import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import h5py

# Chemin du fichier .h5
chemin_results = '/mnt/d/DVPT/DST/results/'
h5_file = chemin_results + 'processed_data_test_with_images.h5'

# Charger les données du fichier HDF5 dans un DataFrame
data = []
with h5py.File(h5_file, 'r') as h5file:
    data_group = h5file['data']
    for key in data_group.keys():
        image = data_group[key][:]
        if 'label' in data_group[key].attrs:
            label = data_group[key].attrs['label']
            data.append({'image': image, 'label': label})
        else:
            print(f"Warning: No label found for image {key}")

df = pd.DataFrame(data)

# Sélectionner un échantillon aléatoire de 20 images
sample_df = df.sample(n=20)

# Afficher les images avec leurs labels
plt.figure(figsize=(20, 10))
for i, (index, row) in enumerate(sample_df.iterrows()):
    image = row['image']
    label = row['label']

    plt.subplot(4, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')

plt.tight_layout()
plt.show()
