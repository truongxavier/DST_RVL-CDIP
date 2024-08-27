"""
Ce fichier Python effectue les opérations suivantes :

1. **Importation des bibliothèques** :
   - Importation des bibliothèques nécessaires pour la manipulation des données, le traitement des images, la journalisation, et la gestion de la concurrence.

2. **Définition des chemins** :
   - Définition des chemins vers les répertoires contenant les images, les étiquettes, les ensembles de données et les résultats.

3. **Définition des dimensions cibles** :
   - Définition des dimensions cibles pour le redimensionnement des images.

4. **Configuration de la journalisation** :
   - Configuration de la journalisation pour enregistrer les messages de log dans un fichier et les afficher dans la console.

5. **Définition de la fonction `collate_fn`** :
   - Définition de la fonction `collate_fn` pour filtrer les éléments nuls dans un lot d'images et séparer les images et les étiquettes.

6. **Définition de la fonction `process_batch`** :
   - Définition de la fonction `process_batch` pour convertir une liste d'images en tableaux numpy et retourner les images et les étiquettes du lot.

7. **Définition de la fonction principale `main`** :
   - Définition de la fonction principale `main` pour configurer et démarrer un cluster Dask local avec des ressources spécifiées.

8. **Définition de la classe `ImageDataset`** :
   - Définition de la classe `ImageDataset` pour charger et traiter les images à partir d'un fichier CSV contenant les chemins des images et leurs étiquettes.

9. **Méthode `__init__` de `ImageDataset`** :
   - Initialisation de la classe avec le fichier CSV, le répertoire racine et la taille des images.

10. **Méthode `__len__` de `ImageDataset`** :
    - Retourne la longueur du DataFrame des étiquettes.

11. **Méthode `__getitem__` de `ImageDataset`** :
    - Charge une image et son étiquette à partir du DataFrame des étiquettes.
    - Redimensionne l'image si la taille est spécifiée.
    - Convertit l'image en tableau numpy.

12. **Méthode `resize_image` de `ImageDataset`** :
    - Redimensionne l'image tout en maintenant le rapport d'aspect en utilisant l'algorithme de rééchantillonnage Lanczos.

13. **Chargement des données et création du DataLoader** :
    - Chargement des données à partir du DataFrame des étiquettes.
    - Création d'un DataLoader pour itérer sur les données en utilisant plusieurs threads.

14. **Traitement des lots d'images** :
    - Traitement des lots d'images en utilisant la fonction `process_batch`.

15. **Sauvegarde des images traitées** :
    - Sauvegarde des images traitées dans un fichier HDF5.

16. **Libération des ressources** :
    - Libération des ressources et nettoyage de la mémoire.
"""

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
from threading import Semaphore, Lock
import time
import tempfile

chemin_images = '/mnt/d/DVPT/projet/images/'
chemin_labels = '/mnt/d/DVPT/projet/labels/'
chemin_datasets = '/mnt/d/DVPT/projet/datasets/'
chemin_results = '/mnt/d/DVPT/projet/results/charged_images/'
csv_final = 'charged_images_'
origin_file_name = 'test'

TARGET_WIDTH = 800
TARGET_HEIGHT = 1000

logging.basicConfig(
    level=logging.INFO,  # Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format du message de logging
    handlers=[
        logging.FileHandler("chargement_images.log"),  # Enregistrer les logs dans un fichier
        logging.StreamHandler()  # Afficher les logs dans la console
    ]
)

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    images, labels = zip(*batch)
    return images, labels

def process_batch(images, batch_labels):
    batch_images = [np.array(image) for image in images]
    return batch_images, batch_labels

def main():
    # Configurer Dask pour utiliser un cluster local avec plus de ressources
    cluster = LocalCluster(n_workers=20, threads_per_worker=4, memory_limit='6GB', dashboard_address=':8788')
    client = Client(cluster)

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
                # if self.image_size:
                #     image = self.resize_image(image)  # Redimensionner l'image
                image = np.array(image)  # Convertir l'image en tableau numpy
            except UnidentifiedImageError:
                logging.info(f"Cannot identify image file {img_name}. Skipping.")
                return None, None
            label = self.labels_df.iloc[idx, 1]
            return image, label

        # def resize_image(self, image):
        #     # Redimensionne l'image tout en maintenant le rapport d'aspect
        #     # en utilisant l'algorithme de rééchantillonnage Lanczos
        #     image = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        #     return image

    # Créer le dataset et le dataloader sans transformation
    dataset = ImageDataset(csv_file=chemin_labels + origin_file_name + '.txt', root_dir=chemin_images) #, image_size=(TARGET_WIDTH, TARGET_HEIGHT))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=6, collate_fn=collate_fn)


    # Exemple de traitement des images par plus petits lots
    futures = []
    batch_size = 10  # Taille des plus petits lots
    max_futures = 8  # Limiter le nombre de futures en attente
    semaphore = Semaphore(max_futures)
    file_counter = 0
    file_lock = Lock()  # Verrou pour les accès aux fichiers

    with ThreadPoolExecutor(max_workers=16) as executor:
        total_batches = len(dataloader)
        with tqdm(total=total_batches, desc="Processing Batches") as pbar:
            for images, batch_labels in dataloader:
                if len(images) == 0:
                    continue  # Skip empty batches
                for i in range(0, len(images), batch_size):
                    small_batch_images = images[i:i + batch_size]
                    small_batch_labels = batch_labels[i:i + batch_size]
                    semaphore.acquire()
                    future = executor.submit(process_batch, small_batch_images, small_batch_labels)
                    futures.append(future)

                    def callback(fut):
                        nonlocal file_counter
                        try:
                            semaphore.release()
                            pbar.update(1)  # Mettre à jour la barre de progression
                            batch_images, batch_labels = fut.result()
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5', dir=chemin_datasets) as temp_file:
                                intermediate_file = temp_file.name

                            with file_lock:  # Utiliser un verrou pour garantir un accès exclusif
                                logging.info(f"Opening file {intermediate_file} for writing")
                                with h5py.File(intermediate_file, 'w') as h5file:
                                    data_group = h5file.create_group('data')
                                    for j, (image, label) in enumerate(zip(batch_images, batch_labels)):
                                        img_dataset_name = f'image_{j}'
                                        data_group.create_dataset(img_dataset_name, data=image)
                                        data_group[img_dataset_name].attrs['label'] = label
                                logging.info(f"Closed file {intermediate_file} after writing")

                            # Log the image and label
                            logging.info(f"Processed batch {file_counter} with {len(batch_images)} images")

                            # Libérer la mémoire utilisée par les images traitées
                            del batch_images
                            gc.collect()
                            file_counter += 1
                        except Exception as e:
                            logging.error(f"Exception in callback: {e}")

                    future.add_done_callback(callback)

            for future in as_completed(futures):
                future.result()  # Assurez-vous que toutes les futures sont terminées

    # Libération des ressources et nettoyage de la mémoire
    client.close()
    cluster.close()
    gc.collect()

    # Fusionner les fichiers intermédiaires en un seul fichier
    output_file = os.path.join(chemin_results, f'{csv_final}{origin_file_name}.h5')
    with file_lock:  # Utiliser un verrou pour garantir un accès exclusif
        logging.info(f"Opening file {output_file} for writing")
        with h5py.File(output_file, 'w') as h5file:
            data_group = h5file.create_group('data')
            for file in os.listdir(chemin_datasets):
                if file.endswith('.h5'):
                    intermediate_file = os.path.join(chemin_datasets, file)
                    logging.info(f"Opening file {intermediate_file} for reading")
                    with h5py.File(intermediate_file, 'r') as intermediate_h5:
                        intermediate_data_group = intermediate_h5['data']
                        for dataset_name in intermediate_data_group:
                            data = intermediate_data_group[dataset_name][:]
                            data_group.create_dataset(dataset_name, data=data)
                    logging.info(f"Closed file {intermediate_file} after reading")
        logging.info(f"Closed file {output_file} after writing")



if __name__ == "__main__":
    logging.info("Starting the program")
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    duration_minutes = round(duration / 60, 2)
    logging.info(f"Program execution time: {duration_minutes} minutes")
