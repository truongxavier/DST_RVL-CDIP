import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np

# Chemins des fichiers
chemin_images = '/mnt/d/DVPT/projet/images/'
chemin_labels = '/mnt/d/DVPT/projet/labels/'
origin_file_name = 'train'

# Charger les labels
df = pd.read_csv(os.path.join(chemin_labels, origin_file_name + '.txt'), sep=" ", header=None, names=['image_chemin', 'label'])

# Filtrer les images avec le label 11
df_label_11 = df[df['label'] == 12]

# Sélectionner un échantillon de 1000 images
sample_size = 100
sample_images = df_label_11.sample(n=sample_size, random_state=42)

# Fonction pour lire une image
def read_image(image_path):
    if not os.path.exists(image_path):
        return np.zeros((1, 1))  # Retourne une image vide si le fichier n'existe pas
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((1, 1))  # Retourne une image vide si l'image ne peut pas être lue
    return image

# Lire les images
images = [read_image(os.path.join(chemin_images, img_path)) for img_path in sample_images['image_chemin']]

# Définir la taille de la grille
grid_size = int(np.ceil(np.sqrt(sample_size)))

# Créer une figure pour afficher les images
fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
axes = axes.flatten()

# Afficher les images dans la grille
for img, ax in zip(images, axes):
    ax.imshow(img, cmap='gray')
    ax.axis('off')

# Supprimer les axes restants
for ax in axes[len(images):]:
    ax.axis('off')

# Sauvegarder la figure
plt.tight_layout()
#plt.savefig(os.path.join(chemin_results, 'sample_label_11.png'))
plt.show()
