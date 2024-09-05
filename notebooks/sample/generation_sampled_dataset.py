import os
import pandas as pd
import random
from datasets import load_from_disk
from tqdm import tqdm  # Importer tqdm pour les barres de progression

# Chemins vers les datasets Hugging Face
train_dataset_path = r"C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\train_dataset_ID_Resize"
val_dataset_path = r"C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\val_dataset_ID_Resize"

# Charger les datasets
train_dataset = load_from_disk(train_dataset_path)
val_dataset = load_from_disk(val_dataset_path)

# Fonction pour échantillonner le dataset de manière équilibrée avec barre de progression
def sample_balanced(dataset, num_samples_per_class):
    label_indices = list(range(dataset.features['label'].num_classes))
    samples = []
    for label_idx in tqdm(label_indices, desc="Échantillonnage par label"):
        label_samples = [i for i, example in enumerate(dataset) if example['label'] == label_idx]
        
        if len(label_samples) < num_samples_per_class:
            print(f"Warning: Requested {num_samples_per_class} samples for label '{label_idx}', but only {len(label_samples)} available. Using all available samples.")
            samples += label_samples
        else:
            samples += random.sample(label_samples, num_samples_per_class)
    
    # Récupérer les IDs d'images correspondant aux échantillons sélectionnés
    sample_ids = [dataset[i]['image_ID'] for i in samples]
    
    return sample_ids

# Paramètres pour l'échantillonnage
num_samples_per_class_train = 1500
num_samples_per_class_val = 400

# Échantillonner les datasets avec barres de progression et récupérer les IDs
print("Échantillonnage du dataset d'entraînement...")
train_ids = sample_balanced(train_dataset, num_samples_per_class_train)

print("Échantillonnage du dataset de validation...")
val_ids = sample_balanced(val_dataset, num_samples_per_class_val)

# Générer les DataFrames avec les IDs récupérés
print("Génération des fichiers CSV...")
train_df = pd.DataFrame(train_ids, columns=['image_id'])
val_df = pd.DataFrame(val_ids, columns=['image_id'])

# Sauvegarder les DataFrames en CSV avec barre de progression
print("Sauvegarde du fichier CSV pour le dataset d'entraînement...")
train_df.to_csv(r'C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\train_ids_sampled_030924.csv', index=False)

print("Sauvegarde du fichier CSV pour le dataset de validation...")
val_df.to_csv(r'C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\val_ids_sampled_030924.csv', index=False)

print("CSV files generated: train_ids_sampled_030924.csv, val_ids_sampled_030924.csv")
