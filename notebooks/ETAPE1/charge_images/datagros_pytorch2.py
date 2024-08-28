import os
import pandas as pd
import numpy as np
import logging
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dask.distributed import Client, LocalCluster
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import gc

chemin_images = '/mnt/d/DVPT/projet/images/'
chemin_labels = '/mnt/d/DVPT/projet/labels/'
chemin_datasets = '/mnt/d/DVPT/projet/datasets/'
chemin_results = '/mnt/d/DVPT/projet/results/'

csv_final = 'final_processed_data_'
origin_file_name = 'train'


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    images, labels = zip(*batch)
    return images, labels

def process_batch(images, batch_labels, dataset):
    batch_images = []
    for image in images:
        batch_images.append(image)
    return batch_images, batch_labels

def main():
    # Configurer Dask pour utiliser un cluster local avec plus de ressources
    cluster = LocalCluster(n_workers=3, threads_per_worker=1, memory_limit='2GB', dashboard_address=':8788')
    client = Client(cluster)



    class ImageDataset(Dataset):
        def __init__(self, csv_file, root_dir, image_size=None):
            self.labels_df = pd.read_csv(csv_file, sep=" ", header=None)
            self.root_dir = root_dir

        def __len__(self):
            return len(self.labels_df)

        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
            try:
                image = Image.open(img_name)
                image = np.array(image)  # Convertir l'image en tableau numpy
            except UnidentifiedImageError:
                logging.info(f"Cannot identify image file {img_name}. Skipping.")
                return None, None
            label = self.labels_df.iloc[idx, 1]
            return image, label

    # Créer le dataset et le dataloader sans transformation
    dataset = ImageDataset(csv_file=chemin_labels + 'test.txt', root_dir=chemin_images)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=3, collate_fn=collate_fn)

    # Créer un fichier de sortie initial
    output_file = chemin_results + 'charged_test_with_images.h5'
    with h5py.File(output_file, 'w') as h5file:
        h5file.create_group('data')

    # Exemple de traitement des images par plus petits lots
    futures = [3]
    batch_size = 100  # Taille des plus petits lots
    with ThreadPoolExecutor() as executor:
        for images, batch_labels in tqdm(dataloader, desc="Processing Batches"):
            if len(images) == 0:
                continue  # Skip empty batches
            for i in range(0, len(images), batch_size):
                small_batch_images = images[i:i + batch_size]
                small_batch_labels = batch_labels[i:i + batch_size]
                futures.append(executor.submit(process_batch, small_batch_images, small_batch_labels, dataset))

            for future in as_completed(futures):
                batch_images, batch_labels = future.result()
                images_list = [np.array(image) for image in batch_images]

                # Ajouter le lot actuel au fichier HDF5
                with h5py.File(output_file, 'a') as h5file:
                    data_group = h5file['data']
                    for j, (image, label) in enumerate(zip(images_list, batch_labels)):
                        img_dataset_name = f'image_{len(data_group)}'
                        data_group.create_dataset(img_dataset_name, data=image)
                        data_group[img_dataset_name].attrs['label'] = label

                # Libérer la mémoire utilisée par les images traitées
                del images_list
                gc.collect()

    # Charger les données du fichier HDF5 dans un DataFrame
    logging.info("Chargement des données dans un DataFrame")
    data = []
    with h5py.File(output_file, 'r') as h5file:
        data_group = h5file['data']
        for key in data_group.keys():
            image = data_group[key][:]
            label = data_group[key].attrs['label']
            data.append({'image': image, 'label': label})
    df = pd.DataFrame(data)

    # Visualisation (exemple)
    logging.info("Début de la visualisation")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.imshow(df.iloc[0]['image'], cmap='gray')
    plt.title(f"Label: {df.iloc[0]['label']}")
    plt.show()

if __name__ == "__main__":
    main()
