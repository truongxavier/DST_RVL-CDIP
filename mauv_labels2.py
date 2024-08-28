import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Définition des chemins des répertoires
labels_dir = "/home/serge/project_DST/env_dstproj/data-rvl_cdip/labels"
images_dir = "/home/serge/project_DST/env_dstproj/data-rvl_cdip/images"
output_dir = "/home/serge/project_DST/env_dstproj/data-rvl_cdip/output_images"
data_dir = "/home/serge/project_DST/env_dstproj/data-rvl_cdip"  # Dossier pour enregistrer les DataFrames

# Dictionnaire des catégories de labels
labels_dict = {
    "0": "letter",
    "1": "form",
    "2": "email",
    "3": "handwritten",
    "4": "advertisement",
    "5": "scientific report",
    "6": "scientific publication",
    "7": "specification",
    "8": "file folder",
    "9": "news article",
    "10": "budget",
    "11": "invoice",
    "12": "presentation",
    "13": "questionnaire",
    "14": "resume",
    "15": "memo"
}

# Fonction pour lire les DataFrames des fichiers de labels
def read_labels_files():
    train_file = os.path.join(labels_dir, "train.txt")
    val_file = os.path.join(labels_dir, "val.txt")
    test_file = os.path.join(labels_dir, "test.txt")

    train_df = pd.read_csv(train_file, sep='\s+', header=None, names=['image_path', 'label'])
    val_df = pd.read_csv(val_file, sep='\s+', header=None, names=['image_path', 'label'])
    test_df = pd.read_csv(test_file, sep='\s+', header=None, names=['image_path', 'label'])
    
    return train_df, val_df, test_df

# Fonction pour échantillonner les images
def sample_images(df, sample_size=200):
    # Créer une liste pour stocker les DataFrames échantillonnés
    sampled_dfs = []
    
    # Grouper par 'label'
    grouped = df.groupby('label')
    
    for label, group in grouped:
        # Échantillonner le groupe actuel
        sampled_group = group.sample(n=min(len(group), sample_size))
        sampled_dfs.append(sampled_group)
    
    # Concaténer les DataFrames échantillonnés
    sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
    
    return sampled_df

# Fonction pour vérifier les images dans le thread principal
def verify_image(row, source_images_dir):
    image_path = row['image_path']
    label = str(row['label'])
    full_image_path = os.path.join(source_images_dir, image_path)
    
    if not os.path.exists(full_image_path):
        print(f"Warning: Image {full_image_path} not found.")
        return None
    
    # Lire l'image avec PIL
    img = Image.open(full_image_path).convert('L')  # Convertir en niveaux de gris
    
    # Convertir en tableau NumPy pour l'affichage
    img_array = np.array(img)
    
    plt.imshow(img_array, cmap='gray')  # Utiliser la carte de couleurs 'gray'
    plt.title(f"Label: {labels_dict.get(label, 'unknown')}")
    plt.axis('off')
    plt.show()  # Bloque l'exécution jusqu'à ce que l'utilisateur ferme la fenêtre

    response = input(f"Is the label '{labels_dict.get(label, 'unknown')}' correct for this image? (yes/no): ").strip().lower()
    plt.close()  # Fermer l'image après l'affichage

    if response != 'yes':
        return {
            'image_path': image_path,
            'label': label
        }
    return None

# Fonction pour traiter les images en série
def process_images(dataframe, source_images_dir):
    incorrect_labels = []
    for _, row in dataframe.iterrows():
        result = verify_image(row, source_images_dir)
        if result:
            incorrect_labels.append(result)
    return pd.DataFrame(incorrect_labels)

# Fonction pour générer un graphique du nombre de mauvais labels par classe
def plot_incorrect_labels(df, dataset_type):
    label_counts = df['label'].value_counts()
    label_names = [labels_dict.get(label, 'unknown') for label in label_counts.index]

    plt.figure(figsize=(12, 6))
    plt.bar(label_names, label_counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Incorrect Labels')
    plt.title(f'Number of Incorrect Labels by Class ({dataset_type.capitalize()})')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'incorrect_labels_{dataset_type}.png'))
    plt.show()

# Fonction principale pour exécuter le processus
def main():
    train_df, val_df, test_df = read_labels_files()
    
    # Échantillonner 200 images par classe
    sampled_train_df = sample_images(train_df)
    sampled_val_df = sample_images(val_df)
    sampled_test_df = sample_images(test_df)
    
    # Vérifier les images de manière séquentielle (pas en parallèle)
    incorrect_train_df = process_images(sampled_train_df, images_dir)
    incorrect_val_df = process_images(sampled_val_df, images_dir)
    incorrect_test_df = process_images(sampled_test_df, images_dir)
    
    # Enregistrer les résultats des images incorrectes
    incorrect_train_df.to_csv(os.path.join(data_dir, 'incorrect_train.csv'), index=False)
    incorrect_val_df.to_csv(os.path.join(data_dir, 'incorrect_val.csv'), index=False)
    incorrect_test_df.to_csv(os.path.join(data_dir, 'incorrect_test.csv'), index=False)
    
    # Générer les graphiques
    plot_incorrect_labels(incorrect_train_df, 'train')
    plot_incorrect_labels(incorrect_val_df, 'val')
    plot_incorrect_labels(incorrect_test_df, 'test')

    print("Le processus de vérification des images est terminé.")

if __name__ == "__main__":
    main()
 