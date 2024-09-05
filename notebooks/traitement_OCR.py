import os
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm
import cv2
import easyocr
import nltk
from PIL import Image, UnidentifiedImageError
from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import pandas as pd
import time
import threading
import torch

#-------------------------------------------------------------------------------
#paramétrage de lancement
#-------------------------------------------------------------------------------
#chemin pour trouver les datasets en enregistrer le csv
results_dir = '/content/drive/MyDrive/formation Datascientest/RVL-CDIP/'
#nom du csv
csv_name ='ocr_results_label_'
#label cible sur lequel on va faire le calcul
LABEL_CIBLE = 11
#choix du datset val train ou test
type_dataset ='train'
# Nombre de threads pour le multithreading
num_threads = 24
#répertoire des OCR par image
temp_dir = '/content/temp'+str(LABEL_CIBLE)+'/'
#suffixe du dataset
dataset_suffixName = '_dataset_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

start_time = time.time()
lock = threading.Lock()

if os.path.exists(temp_dir):
    # Get the current time and format it
    current_time = time.strftime("%Y%m%d-%H%M%S")
    # Rename the existing directory with the current time
    new_temp_dir = temp_dir[:-1] + "_" + current_time + '/'
    os.rename(temp_dir, new_temp_dir)

# Create the temp directory
os.makedirs(temp_dir)

# Configurer TensorFlow pour utiliser le GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU configuré pour l'utilisation.")

# Chemin du fichier CSV pour sauvegarder les résultats OCR
csv_path = results_dir +csv_name + type_dataset + str(LABEL_CIBLE)+'.csv'

# Charger le DataFrame existant si le fichier CSV existe
if os.path.exists(csv_path):
    df_ocr = pd.read_csv(csv_path)
    print(f"Chargement du fichier CSV existant : {csv_path}")
else:
    # Initialiser un DataFrame vide si le fichier n'existe pas
    df_ocr = pd.DataFrame(columns=['image_ID', 'texte_de_ocr', 'confiance_de_ocr', 'coordonnees_du_text',
                                   'coordonnees_des_zones_de_text', 'label_de_image'])
    print("Aucun fichier CSV trouvé, création d'un nouveau DataFrame.")

# Chargement des datasets
dataset = load_from_disk(os.path.join(results_dir, type_dataset + dataset_suffixName))

# Fonction pour filtrer les images du label
def filter_images(index):
    try:
        example = dataset[index]
        if example['label'] == LABEL_CIBLE:
            return example['image'], example['label'], example['image_ID']
    except Exception as e:
        print(f"[WARNING] Erreur lors de la récupération de l'image à l'index {index}: {e}")
    return None, None, None



# Filtrer les indices pour le traitement
indices = list(range(len(dataset)))

# Utiliser ThreadPoolExecutor pour le multithreading avec une barre de progression
print(f"début du filtrage des image sur le label : {LABEL_CIBLE}")
label_invoice_images = []
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(filter_images, i) for i in indices]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Filtrage des images", unit="image"):
        result, label, image_ID= future.result()
        if result is not None:
            label_invoice_images.append((result, label, image_ID))

print(f"Nombre d'images filtrées : {len(label_invoice_images)}")

# Fonction de prétraitement de l'image avec OpenCV
def preprocess_image(image_bytes):
    try:
        # Charger l'image avec PIL pour gérer le format .tif
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')  # Convertir en RGB pour compatibilité OpenCV

        # Convertir l'image en un tableau NumPy
        image_np = np.array(image)

        # Redimensionnement
        image_resized = cv2.resize(image_np, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

        # Seuillage adaptatif
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Suppression du bruit (médian)
        blur = cv2.medianBlur(thresh, 3)

        # Encoder l'image traitée en bytes pour OCR
        _, buffer = cv2.imencode('.png', blur)
        preprocessed_image_bytes = buffer.tobytes()

        return preprocessed_image_bytes

    except Exception as e:
        print(f"Erreur de prétraitement sur l'image: {e}")
        return None

# Initialiser le lecteur EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

# Fonction pour appliquer l'OCR avec EasyOCR
def apply_ocr(image_bytes):
    try:
        # Convertir les bytes en une image pour EasyOCR
        image_np = np.array(Image.open(io.BytesIO(image_bytes)))
        result = reader.readtext(image_np, detail=1)

        # Préparer les données à retourner
        texts = []
        confidences = []
        coordinates = []
        for (coord, text, confidence) in result:
            texts.append(text)
            confidences.append(confidence)
            coordinates.append(coord)

        return texts, confidences, coordinates

    except UnidentifiedImageError:
        print(f"[WARNING] Impossible d'identifier le fichier image. Ignorer...")
        return [], [], []
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'application de l'OCR: {e}")
        return [], [], []

# Liste pour stocker les résultats
ocr_results = []

# Charger les noms d'image déjà traités pour éviter les doublons
images_deja_traitees = set(df_ocr['image_ID'])
print(f"Nombre d'images déjà traitées : {len(images_deja_traitees)}")

# Appliquer le prétraitement et l'OCR sur toutes les images filtrées après prétraitement avec multithreading
print(f"Début de l'OCR sur {len(label_invoice_images)} images")
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {}
    for image, label, image_ID in label_invoice_images:
        # ID de l'image
        image_name = image_ID

        # Vérifier si l'image a déjà été traitée
        if image_name in images_deja_traitees:
            #print(f"Image {image_name} déjà traitée, passage...")
            continue

        # Convertir l'image en bytes directement si ce n'est pas déjà le cas
        if isinstance(image, Image.Image):
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='TIFF')
            image_bytes = image_bytes.getvalue()
        else:
            image_bytes = image

        preprocessed_image_bytes = preprocess_image(image_bytes)
        if preprocessed_image_bytes:
            future = executor.submit(apply_ocr, preprocessed_image_bytes)
            futures[future] = (image_name, label, preprocessed_image_bytes)

    for future in tqdm(as_completed(futures), total=len(futures), desc="Prétraitement et OCR des images", unit="image"):
        image_name, label, preprocessed_image_bytes = futures[future]
        texts, confidences, coordinates = future.result()

        # Enregistrer les résultats dans la liste
        ocr_results.append({
            'image_ID': image_name,
            'texte_de_ocr': " ".join(texts),
            'confiance_de_ocr': confidences,
            'coordonnees_du_text': coordinates,
            'coordonnees_des_zones_de_text': [coordinates],  # Peut être la même que les coordonnées du texte dans ce contexte
            'label_de_image': label
        })

        # Sauvegarder les résultats dans le DataFrame existant et enregistrer le fichier CSV
        new_row = pd.DataFrame([{
            'image_ID': image_name,
            'texte_de_ocr': " ".join(texts),
            'confiance_de_ocr': confidences,
            'coordonnees_du_text': coordinates,
            'coordonnees_des_zones_de_text': [coordinates],  # Peut être la même que les coordonnées du texte dans ce contexte
            'label_de_image': label
        }])
        temp_file = os.path.join(temp_dir, f"temp_{LABEL_CIBLE}_{image_name}.csv")
        new_row.to_csv(temp_file, header=False, index=False)

     # Libérer la mémoire GPU après chaque lot
    torch.cuda.empty_cache()

print("Tous les résultats de l'OCR ont été enregistrés dans "+ csv_path)
end_time = time.time()
processing_time = end_time - start_time
print(f"Temps de traitement total : {round(processing_time / 60, 2)} minutes")
