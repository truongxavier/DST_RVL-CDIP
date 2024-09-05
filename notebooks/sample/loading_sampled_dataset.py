import os
import pandas as pd
from datasets import load_from_disk

# Chemins vers les datasets Hugging Face originaux et les fichiers CSV
train_dataset_path = r"C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\train_dataset_ID_Resize"
val_dataset_path = r"C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\val_dataset_ID_Resize"

train_csv_path = r"C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\train_ids_sampled_030924.csv"
val_csv_path = r"C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\val_ids_sampled_030924.csv"

# Charger les datasets originaux
train_dataset = load_from_disk(train_dataset_path)
val_dataset = load_from_disk(val_dataset_path)

# Charger les CSVs contenant les IDs échantillonnés
train_ids_df = pd.read_csv(train_csv_path)
val_ids_df = pd.read_csv(val_csv_path)

# Convertir les IDs en un set pour une recherche rapide
train_ids_set = set(train_ids_df['image_id'])
val_ids_set = set(val_ids_df['image_id'])

# Obtenir les indices des exemples à sélectionner
train_indices = [i for i, example in enumerate(train_dataset) if example['image_ID'] in train_ids_set]
val_indices = [i for i, example in enumerate(val_dataset) if example['image_ID'] in val_ids_set]

# Sélectionner les indices dans les datasets
reduced_train_dataset = train_dataset.select(train_indices)
reduced_val_dataset = val_dataset.select(val_indices)

# Afficher la taille des datasets réduits
print(f"Taille du reduced_train_dataset : {len(reduced_train_dataset)} exemples")
print(f"Taille du reduced_val_dataset : {len(reduced_val_dataset)} exemples")


# Sauvegarder les datasets réduits
# reduced_train_dataset_path = r"C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\reduced_train_dataset"
# reduced_val_dataset_path = r"C:\Users\Ory-K\Documents\MLE DataScientest\Jul24_bds_extraction\RVL-CDIP_HuggingFace_300824\reduced_val_dataset"

# reduced_train_dataset.save_to_disk(reduced_train_dataset_path)
# reduced_val_dataset.save_to_disk(reduced_val_dataset_path)

# print("Reduced datasets saved successfully!")
