import logging
import os
import cv2
import numpy as np
import base64
import asyncio
import aiofiles
import pytesseract
from scipy.ndimage import generic_filter
from PIL import Image, UnidentifiedImageError
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import nest_asyncio

# Configuration de base du logging
logging.basicConfig(
    level=logging.DEBUG,  # Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format du message de logging
    handlers=[
        logging.FileHandler("debug.log"),  # Enregistrer les logs dans un fichier
        logging.StreamHandler()  # Afficher les logs dans la console
    ]
)

chemin_images = '/mnt/d/DVPT/projet/images/'
chemin_labels = '/mnt/d/DVPT/projet/labels/'
chemin_datasets = '/mnt/d/DVPT/projet/datasets/'
chemin_results = '/mnt/d/DVPT/projet/results/'
csv_final = 'final_processed_data_'
origin_file_name = 'test'

nest_asyncio.apply()

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

def calculate_mean_local_variance(image):
    return generic_filter(image, np.var, size=3).mean()

# def perform_ocr(image):
#     try:
#         if image is None or image.size == 0:
#             return None
#         text = pytesseract.image_to_string(image)
#         return text
#     except Exception as e:
#         logging.error(f"Error performing OCR: {e}")
#         return None

async def process_batch_async(batch):
    try:
        batch['image_chemin'] = batch['image_chemin'].apply(lambda x: logging.info(f"Processing image: {x}") or x)
        batch['image'] = await asyncio.gather(*[read_image_async(chemin_images + path) for path in batch['image_chemin']])
        batch['width'] = batch['image'].apply(lambda x: x.shape[1] if x is not None else None)
        batch['height'] = batch['image'].apply(lambda x: x.shape[0] if x is not None else None)
        batch['dpi'] = batch['image'].apply(lambda x: Image.fromarray(x).info.get('dpi') if x is not None else None)
        batch['blur'] = batch['image'].apply(lambda x: cv2.Laplacian(x, cv2.CV_64F).var() if x is not None else None)
        batch['brightness'] = batch['image'].apply(lambda x: np.mean(x) if x is not None else None)
        batch['contrast'] = batch['image'].apply(lambda x: np.std(x) if x is not None else None)
        batch['entropy'] = batch['image'].apply(lambda x: calculate_entropy(x) if x is not None else None)
        batch['mean_luminance'] = batch['image'].apply(lambda x: calculate_mean_luminance(x) if x is not None else None)
        batch['mean_local_var'] = batch['image'].apply(lambda x: calculate_mean_local_variance(x) if x is not None else None)
        #batch['ocr_text'] = batch['image'].apply(lambda x: perform_ocr(x) if x is not None else None)
        batch.drop('image', axis=1, inplace=True)  # Remove the 'image' column
        return batch
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return batch

def process_batch(batch):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(process_batch_async(batch))

def main():
    try:
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
        npartitions = (total_rows // 1000) + (1 if total_rows % 1000 != 0 else 0)
        logging.info(f"Nombre de partitions : {npartitions}")
        # Repartitionner le DataFrame
        df = df.repartition(npartitions=npartitions)

        # Définir le meta pour indiquer à Dask le type de sortie attendu
        meta = {
            'image_chemin': 'str',
            'label': 'int',
            'width': 'int',
            'height': 'int',
            'dpi': 'object',
            'blur': 'float',
            'brightness': 'float',
            'contrast': 'float',
            'entropy': 'float',
            'mean_luminance': 'float',
            'mean_local_var': 'float',
            'ocr_text': 'object'
        }

        # Traiter les données par lots
        df = df.map_partitions(process_batch, meta=meta)

        # Sauvegarder le DataFrame final
        df_final = df.compute()
        df_final.to_csv(chemin_results + csv_final, index=False)
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
