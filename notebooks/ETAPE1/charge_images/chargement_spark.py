import os
import h5py
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
from PIL import Image

# Initialiser une session Spark
spark = SparkSession.builder \
    .appName("ImageLoader") \
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

# Lire les images et extraire les données
data = []
for index, row in labels_df.iterrows():
    img_path = os.path.join(chemin_images, row['image_name'])
    try:
        with Image.open(img_path) as img:
            image_data = img.tobytes()
            data.append((row['image_name'], image_data, row['label']))
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

# Créer un DataFrame Pandas
df = pd.DataFrame(data, columns=['image_name', 'image_data', 'label'])

# Convertir le DataFrame Pandas en DataFrame Spark
schema = StructType([
    StructField("image_name", StringType(), True),
    StructField("image_data", BinaryType(), True),
    StructField("label", StringType(), True)
])

spark_df = spark.createDataFrame(df, schema=schema)

# Sauvegarder le DataFrame en utilisant Spark
output_path = os.path.join(chemin_results, 'images_dataframe.parquet')
spark_df.write.parquet(output_path)

# Fermer la session Spark
spark.stop()
