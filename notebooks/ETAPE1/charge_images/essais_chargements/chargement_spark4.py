import os
import cv2
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
from tqdm import tqdm
import time
import gc

# Initialiser une session Spark avec des options supplémentaires
spark = SparkSession.builder \
    .appName("ImageLoader") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
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
batch_size = 100  # Taille du lot augmentée
total_images = len(labels_df)
start_time = time.time()

for start in range(0, total_images, batch_size):
    end = min(start + batch_size, total_images)
    batch_df = labels_df.iloc[start:end]
    data = []

    for index, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Traitement des images"):
        img_path = os.path.join(chemin_images, row['image_name'])
        try:
            img = cv2.imread(img_path)
            if img is not None:
                image_data = cv2.imencode('.jpg', img)[1].tobytes()
                data.append((row['image_name'], image_data, row['label']))
            else:
                print(f"Error loading image {img_path}: Image is None")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    # Convertir les données en DataFrame Spark
    schema = StructType([
        StructField("image_name", StringType(), True),
        StructField("image_data", BinaryType(), True),
        StructField("label", StringType(), True)
    ])
    spark_df = spark.createDataFrame(data, schema)
    spark_df.write.mode('overwrite').parquet(os.path.join(chemin_results, f"{csv_final}{start//batch_size}.parquet"))

    # Libérer la mémoire
    del data
    gc.collect()

    # Calculer le temps restant estimé
    elapsed_time = time.time() - start_time
    images_processed = end
    images_remaining = total_images - images_processed
    time_per_image = elapsed_time / images_processed
    estimated_time_remaining = time_per_image * images_remaining
    tqdm.write(f"Images traitées: {images_processed}/{total_images}, Temps restant estimé: {round(estimated_time_remaining/60,2):.2f} minutes")

# Réunir tous les fichiers en un seul DataFrame
all_files = [os.path.join(chemin_results, f) for f in os.listdir(chemin_results) if f.startswith(csv_final)]
combined_df = spark.read.parquet(*all_files)

# Enregistrer le DataFrame combiné
combined_df.write.mode('overwrite').parquet(os.path.join(chemin_results, f"{csv_final}combined.parquet"))
