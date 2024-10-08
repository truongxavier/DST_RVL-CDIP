import os
import pandas as pd
import numpy as np
import logging
#import h5py
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
from scipy.ndimage import generic_filter
#import pytesseract
import io

logging.basicConfig(
    level=logging.INFO,  # Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format du message de logging
    handlers=[
        logging.FileHandler("datagros_plusrapide.log"),  # Enregistrer les logs dans un fichier
        logging.StreamHandler()  # Afficher les logs dans la console
    ]
)

chemin_images = '/mnt/d/DVPT/projet/images/'
chemin_labels = '/mnt/d/DVPT/projet/labels/'
chemin_datasets = '/mnt/d/DVPT/projet/datasets/'
chemin_results = '/mnt/d/DVPT/projet/results/'
csv_final = 'final_processed_data_'
origin_file_name = 'train'

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

def calculate_entropy(image):
    try:
        if image is None or image.size == 0:
            return None
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / hist.sum()  # Normalize the histogram
        entropy = -np.sum(hist * np.log2(hist + 1e-7))  # Calculate entropy
        return entropy
    except Exception as e:
        logging.error(f"Error calculating entropy: {e}")
        return None

def calculate_mean_luminance(image):
    gray_array = np.array(image)
    return np.mean(gray_array)

# def local_variance(arr):
#     return arr.var()

# def calculate_mean_local_variance(image):
#     gray_array = np.array(image)
#     local_var = generic_filter(gray_array, local_variance, size=3)
#     mean_local_var = np.mean(local_var)
#     return mean_local_var

# def perform_ocr(image):
#     try:
#         if image is None or image.size == 0:
#             return None
#         text = pytesseract.image_to_string(image)
#         return text
#     except Exception as e:
#         logging.error(f"Error performing OCR: {e}")
#         return None

async def calculate_dpi_async(image_path):
    try:
        async with aiofiles.open(image_path, 'rb') as f:
            # Lire le fichier en mémoire
            image_data = await f.read()
            # Ouvrir l'image de manière synchrone
            with Image.open(io.BytesIO(image_data)) as img:
                dpi = img.info.get('dpi', (0, 0))
                dpi = (float(dpi[0]), float(dpi[1]))  # Convert to float
                return dpi[0]
    except Exception as e:
        logging.error(f"Error calculating DPI: {e}")
        return None

def calculate_pixel_count(image):
    try:
        if image is None or image.size == 0:
            return None
        return image.size
    except Exception as e:
        logging.error(f"Error calculating pixel count: {e}")
        return None


async def process_batch_async(batch):
    batch['image'] = await asyncio.gather(*[read_image_async(chemin_images + path) for path in batch['image_chemin']])
    batch['width'] = batch['image'].apply(lambda x: x.shape[1] if x is not None else None)
    batch['height'] = batch['image'].apply(lambda x: x.shape[0] if x is not None else None)
    dpi_results = await asyncio.gather(*[calculate_dpi_async(chemin_images + path) for path in batch['image_chemin']])
    batch['dpi'] = dpi_results
    batch['blur'] = batch['image'].apply(lambda x: cv2.Laplacian(x, cv2.CV_64F).var() if x is not None else None)
    batch['brightness'] = batch['image'].apply(lambda x: np.mean(x) if x is not None else None)
    batch['contrast'] = batch['image'].apply(lambda x: np.std(x) if x is not None else None)
    batch['entropy'] = batch['image'].apply(lambda x: calculate_entropy(x) if x is not None else None)
    batch['mean_luminance'] = batch['image'].apply(lambda x: calculate_mean_luminance(x) if x is not None else None)
    #batch['mean_local_var'] = batch['image'].apply(lambda x: calculate_mean_local_variance(x) if x is not None else None)
    #batch['ocr_text'] = batch['image'].apply(lambda x: perform_ocr(x) if x is not None else None)
    batch['pixel_count'] = batch['image'].apply(lambda x: calculate_pixel_count(x) if x is not None else None)
    batch.drop('image', axis=1, inplace=True)  # Remove the 'image' column
    return batch

def process_batch(batch):
    return asyncio.run(process_batch_async(batch))

def main():
    # Configurer Dask pour utiliser un cluster local avec plus de ressources
    cluster = LocalCluster(n_workers=20, threads_per_worker=4, memory_limit='8GB', dashboard_address=':8788')
    client = Client(cluster)

    # Charger le DataFrame
    df = dd.read_csv(chemin_labels + origin_file_name + '.txt', sep=" ", header=None, names=['image_chemin', 'label'])
    #df['image_chemin'] = chemin_images + df['image_chemin']

    # Repartitionner les données en paquets plus petits pour une meilleure parallélisation
    logging.info("Repartitionnement des données")
    # Calculer le nombre total de lignes
    total_rows = df.shape[0].compute()
    logging.info(f"Nombre total de lignes : {total_rows}")
    # Calculer le nombre de partitions nécessaires pour avoir 1000 lignes par partition
    npartitions = (total_rows // 500) + (1 if total_rows % 500 != 0 else 0)
    logging.info(f"Nombre de partitions : {npartitions}")
    # Repartitionner le DataFrame
    df = df.repartition(npartitions=npartitions)

    # Traiter les données par lots avec une barre de progression
    logging.info("Début du traitement des partitions")
    with tqdm(total=df.npartitions, desc="Processing Partitions") as pbar:
        def update_pbar(future):
            pbar.update()
        meta={  'image_chemin': 'object',
                'label': 'int64',
                'width': 'int64',
                'height': 'int64',
                'dpi': 'int64',
                'blur': 'float64',
                'brightness': 'float64',
                'contrast': 'float64',
                'entropy': 'float64',
                'mean_luminance': 'float64',
                'pixel_count': 'int64'
                #'ocr_text': 'object'
                #'mean_local_var': 'float64'
                }
        futures = df.map_partitions(process_batch, meta=meta ).to_delayed()
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
    all_files = [os.path.join(chemin_datasets, f) for f in os.listdir(chemin_datasets) if f.endswith('.csv')]
    df_final = pd.concat([pd.read_csv(file) for file in all_files])
    df_final.to_csv(chemin_results + csv_final + origin_file_name + ".csv", index=False)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Temps de traitement total : {round(processing_time / 60, 2)} minutes")
