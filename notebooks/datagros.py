import time
import dask.dataframe as dd
import cv2
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError
from tqdm import tqdm
from dask.distributed import Client, LocalCluster, progress
import logging
import ipywidgets as widgets

def main():
    # Configurer les logs détaillés
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("dask_processing.log"),
                            logging.StreamHandler()
                        ])

    # Configurer Dask pour utiliser un cluster local avec plus de ressources
    cluster = LocalCluster(n_workers=12, threads_per_worker=2, memory_limit='8GB', dashboard_address=':8788')
    client = Client(cluster)

    chemin_images = '/mnt/d/DVPT/DST/images/'
    chemin_labels = '/mnt/d/DVPT/DST/labels/'
    chemin_datasets = '/mnt/d/DVPT/DST/datasets/'
    chemin_results = '/mnt/d/DVPT/DST/results/'

    # Lire le fichier de labels en utilisant dask
    logging.info("Lecture du fichier de labels")
    df = dd.read_csv(chemin_labels + 'test.txt', sep=' ', names=['image_chemin', 'label'])
    df['image_chemin'] = chemin_images + df['image_chemin']

    def read_image(file_path, *args, **kwargs):
        try:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                return image
            else:
                return np.zeros((1, 1))  # Retourne une image vide si la lecture échoue
        except (FileNotFoundError, UnidentifiedImageError):
            return np.zeros((1, 1))  # Retourne une image vide si une exception est levée

    def process_batch(batch):
        batch['image'] = batch['image_chemin'].apply(read_image, meta=('image', 'object'))
        batch['shape'] = batch['image'].apply(lambda x: x.shape if x is not None else None)
        return batch

    # Repartitionner les données en paquets plus petits pour une meilleure parallélisation
    logging.info("Repartitionnement des données")
    # Calculer le nombre total de lignes
    total_rows = df.shape[0].compute()
    logging.info(f"Nombre total de lignes : {total_rows}")
    # Calculer le nombre de partitions nécessaires pour avoir 100 lignes par partition
    npartitions = (total_rows // 1000) + (1 if total_rows % 100 != 0 else 0)
    logging.info(f"Nombre de partitions : {npartitions}")
    # Repartitionner le DataFrame
    df = df.repartition(npartitions=npartitions)

    # Traiter les données par lots avec une barre de progression
    logging.info("Début du traitement des partitions")
    with tqdm(total=df.npartitions, desc="Processing Partitions") as pbar:
        def update_pbar(future):
            pbar.update()

        futures = df.map_partitions(process_batch, meta={'image_chemin': 'object', 'label': 'int64', 'image': 'object', 'shape': 'object'}).to_delayed()
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
            progress(futures)

    # Lire les fichiers CSV intermédiaires et les concaténer en un seul DataFrame
    logging.info("Lecture des fichiers CSV intermédiaires")
    processed_files = [chemin_datasets + f'processed_batch_{i}.csv' for i in range(len(futures))]
    df = dd.read_csv(processed_files)

    # Convertir en DataFrame pandas pour les opérations de visualisation
    logging.info("Conversion en DataFrame pandas")
    df = df.compute()
    df.to_csv(chemin_results + 'processed_data_test.csv', index=False)

    # Visualisation
    logging.info("Début de la visualisation")
    plt.figure(figsize=(8, 5))
    plt.imshow(df['image'].iloc[1], cmap='gray')
    plt.show()

    df['label'].value_counts().head(15).plot(kind='bar')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Distribution of Labels')
    plt.savefig('distribution_labels.png')
    plt.show()

    df['shape'].value_counts().plot(kind='bar')
    plt.xlabel('Shape')
    plt.ylabel('Count')
    plt.title('Distribution of Shapes')
    plt.show()

if __name__ == '__main__':
    main()
