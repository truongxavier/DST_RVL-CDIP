import os
import h5py
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
from PIL import Image
from tqdm import tqdm
import time
import gc

# Initialiser une session Spark avec des options supplémentaires
spark = SparkSession.builder \
    .appName("ImageLoader") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Chemins des fichiers
chemin_images = '/mnt/d/DVPT/projet/images/'
chemin_labels = '/mnt/d/DVPT/projet/labels/'
chemin_datasets = '/mnt/d/DVPT/projet/datasets/'
chemin_results = '/mnt/d/DVPT/projet/results/charged_images/'
csv_final = 'charged_images_'
origin_file_name = 'test'

# Lire le fichier CSV pour obtenir les chemins des images et leurs labels
csv_file = os.path.join(chemin_labels, origin_file_name + '.txt')
labels_df = pd.read_csv(csv_file, sep=" ", header=None, names=['image_name', 'label'])

# Lire les images et extraire les données par lots
batch_size = 100  # Taille du lot
total_images = len(labels_df)
start_time = time.time()

for start in range(0, total_images, batch_size):
    end = min(start + batch_size, total_images)
    batch_df = labels_df.iloc[start:end]
    data = []

    for index, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Traitement des images"):
        img_path = os.path.join(chemin_images, row['image_name'])
        try:
            with Image.open(img_path) as img:
                image_data = img.tobytes()
                data.append((row['image_name'], image_data, row['label']))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    # Convertir les données en DataFrame Spark
    schema = StructType([
        StructField("image_name", StringType(), True),
        StructField("image_data", BinaryType(), True),
        StructField("label", StringType(), True)
    ])
    spark_df = spark.createDataFrame(data, schema)
    spark_df.write.mode('append').parquet(os.path.join(chemin_results, f"{csv_final}{start//batch_size}.parquet"))

    # Libérer la mémoire
    del data
    gc.collect()

    # Calculer le temps restant estimé
    elapsed_time = time.time() - start_time
    images_processed = end
    images_remaining = total_images - images_processed
    time_per_image = elapsed_time / images_processed
    estimated_time_remaining = time_per_image * images_remaining
    tqdm.write(f"Images traitées: {images_processed}/{total_images}, Temps restant estimé: {estimated_time_remaining:.2f} secondes")
