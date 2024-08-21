from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from dask.distributed import Client, LocalCluster
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import logging  # Ajouter l'importation du module logging

# Définir le dataset personnalisé
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, image_size=None):
        self.labels_df = pd.read_csv(csv_file, sep=" ", header=None)
        self.root_dir = root_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        try:
            image = Image.open(img_name)
            if self.image_size:
                image = self.resize_image(image, self.image_size)  # Redimensionner l'image
            image = np.array(image)  # Convertir l'image en tableau numpy
        except UnidentifiedImageError:
            logging.info(f"Cannot identify image file {img_name}. Skipping.")
            return None, None
        label = self.labels_df.iloc[idx, 1]
        return image, label

    def resize_image(self, image, size):
        # Redimensionner l'image tout en maintenant le rapport d'aspect
        image.thumbnail(size, Image.LANCZOS)
        return image

def collate_fn(batch):
    # Filtrer les éléments None
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    images, labels = zip(*batch)
    return torch.stack([torch.tensor(img) for img in images]), torch.tensor(labels)

def get_max_image_size(csv_file, root_dir):
    labels_df = pd.read_csv(csv_file, sep=" ", header=None)
    max_width, max_height = 0, 0

    for idx in tqdm(range(len(labels_df)), desc="Calculating max image size"):
        img_name = os.path.join(root_dir, labels_df.iloc[idx, 0])
        try:
            with Image.open(img_name) as img:
                width, height = img.size
                if width > max_width:
                    max_width = width
                if height > max_height:
                    max_height = height
        except UnidentifiedImageError:
            logging.info(f"Cannot identify image file {img_name}. Skipping.")

    return max_width, max_height

def main():
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler("dask_processing.log"),
        logging.StreamHandler()
    ])

    # Configurer Dask pour utiliser un cluster local avec plus de ressources
    cluster = LocalCluster(n_workers=20, threads_per_worker=4, memory_limit='8GB', dashboard_address=':8788')
    client = Client(cluster)

    chemin_images = '/mnt/d/DVPT/DST/images/'
    chemin_labels = '/mnt/d/DVPT/DST/labels/'
    chemin_datasets = '/mnt/d/DVPT/DST/datasets/'
    chemin_results = '/mnt/d/DVPT/DST/results/'

    # Déterminer la taille maximale des images
    max_width, max_height = get_max_image_size(chemin_labels + 'test.txt', chemin_images)
    image_size = (max_width, max_height)
    logging.info(f"Max image size: {image_size}")

    # Créer le dataset et le dataloader sans transformation
    dataset = ImageDataset(csv_file=chemin_labels + 'test.txt', root_dir=chemin_images, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=12, collate_fn=collate_fn)

    # Listes pour stocker les chemins des images et les labels
    image_paths = []
    labels = []

    # Exemple de traitement des images par lots
    for images, batch_labels in tqdm(dataloader, desc="Processing Batches"):
        if images.size(0) == 0:
            continue  # Skip empty batches

        # Convertir les images en numpy arrays et stocker les chemins et les labels
        for i in range(len(batch_labels)):
            image_paths.append(dataset.labels_df.iloc[i, 0])
            labels.append(batch_labels[i].item())

    # Créer le DataFrame final
    df = pd.DataFrame({'image_chemin': image_paths, 'label': labels})
    df.to_pickle(chemin_results + 'processed_data_test.pkl')

    # Visualisation (exemple)
    logging.info("Début de la visualisation")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.imshow(dataset[0][0], cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
