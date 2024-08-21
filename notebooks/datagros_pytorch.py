from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from dask.distributed import Client, LocalCluster
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import logging  # Ajouter l'importation du module logging
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_WIDTH = 2000  # Largeur cible pour le redimensionnement
TARGET_HEIGHT = 1000  # Hauteur cible pour le redimensionnement

# Définir le dataset personnalisé
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, image_size=None):
        self.labels_df = pd.read_csv(csv_file, sep=" ", header=None)
        self.root_dir = root_dir
        self.image_size = image_size  # Ajouter l'attribut image_size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Cette fonction est utilisée pour obtenir un échantillon spécifique du dataset personnalisé.
        # Elle prend un index en argument et retourne l'image correspondante et son label.
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        try:
            image = Image.open(img_name)
            if self.image_size:
                image = self.resize_image(image)  # Redimensionner l'image
            image = np.array(image)  # Convertir l'image en tableau numpy
        except UnidentifiedImageError:
            logging.info(f"Cannot identify image file {img_name}. Skipping.")
            return None, None
        label = self.labels_df.iloc[idx, 1]
        return image, label

    def resize_image(self, image):
        # Redimensionne l'image tout en maintenant le rapport d'aspect
        # en utilisant l'algorithme de rééchantillonnage Lanczos
        image = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        return image

# Cette fonction collate_fn est utilisée comme argument collate_fn dans le DataLoader.
# Son rôle est de traiter un batch d'échantillons du dataset personnalisé.
# Elle filtre les éléments None, puis extrait les images et les labels du batch.
# Ensuite, elle convertit les images en tenseurs torch et renvoie le batch de tenseurs d'images et de tenseurs de labels.
# Si le batch est vide, elle renvoie des tenseurs vides.
def collate_fn(batch):
    # Filtrer les éléments None
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    images, labels = zip(*batch)
    return torch.stack([torch.tensor(img) for img in images]), torch.tensor(labels)

def process_batch(images, batch_labels, dataset):
    image_paths = []
    labels = []
    for i in range(len(batch_labels)):
        image_paths.append(dataset.labels_df.iloc[i, 0])
        labels.append(batch_labels[i].item())
        
    return image_paths, labels

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

    # Créer le dataset et le dataloader sans transformation
    dataset = ImageDataset(csv_file=chemin_labels + 'test.txt', root_dir=chemin_images, image_size=(TARGET_WIDTH, TARGET_HEIGHT))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=12, collate_fn=collate_fn)

    # Listes pour stocker les chemins des images et les labels
    image_paths = []
    labels = []

    # Exemple de traitement des images par lots
    futures = []
    with ThreadPoolExecutor() as executor:
        for images, batch_labels in tqdm(dataloader, desc="Processing Batches"):
            if images.size(0) == 0:
                continue  # Skip empty batches
            futures.append(executor.submit(process_batch, images, batch_labels, dataset))

        for future in as_completed(futures):
            batch_image_paths, batch_labels = future.result()
            image_paths.extend(batch_image_paths)
            labels.extend(batch_labels)

    # Créer le DataFrame final
    df = pd.DataFrame({'image_chemin': image_paths, 'label': labels})
    df.to_pickle(chemin_results + 'processed_data_test.pkl')

    # Visualisation (exemple)
    logging.info("Début de la visualisation")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.imshow(dataset[1][0], cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
