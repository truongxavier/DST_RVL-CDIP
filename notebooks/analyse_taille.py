import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import seaborn as sns
from tqdm import tqdm
import pytesseract
import dask.dataframe as dd
from PIL import Image
import shutil

# Chemins des fichiers et répertoires
csv_final = 'final_processed_data.csv'
chemin_images = '/mnt/d/DVPT/projet/images/'
chemin_images_resized = '/mnt/d/DVPT/projet/images_resized/'
chemin_datasets = '/mnt/d/DVPT/projet/datasets/'
chemin_results = '/mnt/d/DVPT/projet/results/'


# Créer le répertoire pour les images redimensionnées s'il n'existe pas
# all_files = [os.path.join(chemin_datasets, f) for f in os.listdir(chemin_datasets) if f.endswith('.csv')]
# df_final = pd.concat([pd.read_csv(file) for file in all_files])
# df_final.to_csv(chemin_results + csv_final, index=False)


os.makedirs(chemin_images_resized, exist_ok=True)

# Charger le CSV final dans un DataFrame
df = pd.read_csv(chemin_results + csv_final)
# Afficher le nombre d'images dans le DataFrame
num_images = len(df)
print(f"Nombre d'images : {num_images}")

# Visualiser 10 images
def visualize_images(df, num_images=10):
    sample_df = df.sample(num_images)
    plt.figure(figsize=(20, 10))
    for i, row in enumerate(sample_df.itertuples()):
        image_path = os.path.join(chemin_images, row.image_chemin)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            plt.subplot(2, 5, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Label: {row.label}")
            plt.axis('off')
    plt.show()

# Étude sur la distribution de la largeur (width)
def study_width_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['width'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Image Widths')
    plt.xlabel('Width')
    plt.ylabel('Frequency')
    plt.xlim(500, 1000)  # Rescale the x-axis
    plt.show()

def study_height_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['height'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Image Heights')
    plt.xlabel('Height')
    plt.ylabel('Frequency')
    plt.xlim(950, 1050)  # Rescale the x-axis
    plt.show()

# Tester différentes tailles d'image et enregistrer les images redimensionnées
def resize_images(df, sizes=[(1000, 1000), (1500, 1500), (2000, 2000), (2500, 2500)], sample_size=50):
    sample_df = df.sample(sample_size)
    resized_image_paths = {}
    original_images_dir = os.path.join(chemin_images_resized, "originals")
    os.makedirs(original_images_dir, exist_ok=True)

    # Add the originals directory to resized_image_paths
    resized_image_paths["originals"] = []

    for size in sizes:
        size_dir = os.path.join(chemin_images_resized, f"{size[0]}x{size[1]}")
        os.makedirs(size_dir, exist_ok=True)
        resized_image_paths[size] = []

        for row in tqdm(sample_df.itertuples(), desc=f"Resizing to {size[0]}x{size[1]}"):
            image_path = os.path.join(chemin_images, row.image_chemin)
            resized_image_path = os.path.join(size_dir, os.path.basename(row.image_chemin))
            original_image_copy_path = os.path.join(original_images_dir, os.path.basename(row.image_chemin))

            # Copy the original image if it doesn't already exist in the originals directory
            if not os.path.exists(original_image_copy_path):
                shutil.copy(image_path, original_image_copy_path)
                resized_image_paths["originals"].append(original_image_copy_path)

            # Check if the resized image already exists
            if os.path.exists(resized_image_path):
                resized_image_paths[size].append(resized_image_path)
                continue

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized_image = cv2.resize(image, size)
                cv2.imwrite(resized_image_path, resized_image)
                resized_image_paths[size].append(resized_image_path)

    return resized_image_paths

def perform_ocr(resized_image_paths):
    ocr_results = {}
    for size, paths in resized_image_paths.items():
        ocr_results[size] = []
        for image_path in tqdm(paths, desc=f"Performing OCR on {size[0]}x{size[1]} images"):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                ocr_text = " ".join(ocr_data['text'])
                ocr_confidences = [conf for conf in ocr_data['conf'] if conf != -1]
                avg_confidence = np.mean(ocr_confidences) if ocr_confidences else 0
                # Get DPI information
                with Image.open(image_path) as img:
                    dpi = img.info.get('dpi', (0, 0))
                    dpi = (float(dpi[0]), float(dpi[1]))  # Convert to float
                ocr_results[size].append((ocr_text, avg_confidence, dpi))
    return ocr_results

# Évaluer les résultats OCR
def evaluate_ocr_results(ocr_results):
    for size, results in ocr_results.items():
        total_chars = sum(len(text) for text, _, _ in results)
        avg_chars = total_chars / len(results) if results else 0
        total_words = sum(len(text.split()) for text, _, _ in results)
        avg_words = total_words / len(results) if results else 0
        alnum_chars = sum(sum(c.isalnum() for c in text) for text, _, _ in results)
        avg_alnum_chars = alnum_chars / len(results) if results else 0
        avg_confidence = np.mean([conf for _, conf, _ in results]) if results else 0
        avg_dpi = np.mean([dpi[0] for _, _, dpi in results]) if results else 0
        print(f"Size {size}:")
        print(f"  Average OCR text length: {avg_chars}")
        print(f"  Average number of words: {avg_words}")
        print(f"  Average number of alphanumeric characters: {avg_alnum_chars}")
        print(f"  Average OCR confidence: {avg_confidence}")
        print(f"  Average DPI: {avg_dpi}")

if __name__ == "__main__":
    # Visualiser 10 images
    visualize_images(df)

    # Étude sur la distribution de la largeur (width)
    study_width_distribution(df)

    study_height_distribution(df)

    # Tester différentes tailles d'image et enregistrer les images redimensionnées
    resized_image_paths = resize_images(df)
    ocr_results = perform_ocr(resized_image_paths)

    # # Évaluer les résultats OCR
    evaluate_ocr_results(ocr_results)
