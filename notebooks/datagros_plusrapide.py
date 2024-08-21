import os
import pandas as pd
import numpy as np
import logging
import h5py
import cv2
import base64
import dask.dataframe as dd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dask.distributed import Client, LocalCluster
from PIL import Image, UnidentifiedImageError
import time
import asyncio
import aiofiles
import aiohttp

async def read_image_async(image_path):
    if not os.path.exists(image_path):
        logging.warning(f"File not found: {image_path}")
        return np.zeros((1, 1))  # Retourne une image vide si le fichier n'existe pas
    try:
        async with aiofiles.open(image_path, 'rb') as f:
            image_data = await f.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError
        return image
    except (FileNotFoundError, UnidentifiedImageError):
        logging.warning(f"Cannot read image: {image_path}")
        return np.zeros((1, 1))  # Retourne une image vide si une exception est levée

def serialize_image(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def image_to_list(image):
    return image.flatten().tolist()

async def process_batch_async(batch):
    batch['image'] = await asyncio.gather(*[read_image_async(path) for path in batch['image_chemin']])
    batch['width'] = [img.shape[1] if img is not None else None for img in batch['image']]
    batch['height'] = [img.shape[0] if img is not None else None for img in batch['image']]
    batch.drop('image', axis=1, inplace=True)  # Remove the 'image' column
    return batch

def process_batch(batch):
    return asyncio.run(process_batch_async(batch))

def main():
    # Configurer Dask pour utiliser un cluster local avec plus de ressources
    cluster = LocalCluster(n_workers=20, threads_per_worker=4, memory_limit='8GB', dashboard_address=':8788')
    client = Client(cluster)

    chemin_images = '/mnt/d/DVPT/projet/images/'
    chemin_labels = '/mnt/d/DVPT/projet/labels/'
    chemin_datasets = '/mnt/d/DVPT/projet/datasets/'
    chemin_results = '/mnt/d/DVPT/projet/results/'

    # Charger le DataFrame
    df = dd.read_csv(chemin_labels + 'test.txt', sep=" ", header=None, names=['image_chemin', 'label'])
    df['image_chemin'] = chemin_images + df['image_chemin']

    # Repartitionner les données en paquets plus petits pour une meilleure parallélisation
    logging.info("Repartitionnement des données")
    # Calculer le nombre total de lignes
    total_rows = df.shape[0].compute()
    logging.info(f"Nombre total de lignes : {total_rows}")
    # Calculer le nombre de partitions nécessaires pour avoir 1000 lignes par partition
    npartitions = (total_rows // 1000) + (1 if total_rows % 1000 != 0 else 0)
    logging.info(f"Nombre de partitions : {npartitions}")
    # Repartitionner le DataFrame
    df = df.repartition(npartitions=npartitions)

    # Traiter les données par lots avec une barre de progression
    logging.info("Début du traitement des partitions")
    with tqdm(total=df.npartitions, desc="Processing Partitions") as pbar:
        def update_pbar(future):
            pbar.update()

        futures = df.map_partitions(process_batch, meta={'image_chemin': 'object', 'label': 'int64', 'width': 'int64', 'height': 'int64'}).to_delayed()
        with ThreadPoolExecutor(max_workers=16) as executor:
            for i, future in enumerate(futures):
                future.add_done_callback(update_pbar)
                start_time = time.time()
                result = dd.compute(future)[0]
                end_time = time.time()
                processing_time = end_time - start_time
                result.to_csv(chemin_datasets + f'processed_batch_{i}.csv', index=False)
                logging.info(f"Batch {i} traité et sauvegardé. Temps de traitement : {round(processing_time,2)} secondes")
                estimated_remaining_time = round((processing_time * (df.npartitions - i - 1)) / 60, 2)
                logging.info(f"Temps de traitement restant estimé : {estimated_remaining_time} minutes")
                pbar.update(1)  # Mettre à jour la barre de progression

    # Lire les fichiers CSV intermédiaires et les concaténer en un seul DataFrame
    logging.info("Lecture des fichiers CSV intermédiaires")
    all_files = [os.path.join(chemin_datasets, f) for f in os.listdir(chemin_datasets) if f.startswith('processed_batch_')]
    df_final = dd.concat([dd.read_csv(file) for file in all_files])
    df_final.to_csv(chemin_results + 'final_processed_data.csv', index=False)

if __name__ == "__main__":
    main()
