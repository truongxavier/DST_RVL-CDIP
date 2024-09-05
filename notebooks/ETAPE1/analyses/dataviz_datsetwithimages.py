import os
from datasets import load_from_disk
import matplotlib.pyplot as plt

# Chemin pour charger les datasets
results_dir = '/mnt/d/DVPT/projet/data/'

# Charger les datasets enregistrés
train_dataset = load_from_disk(os.path.join(results_dir, 'train_dataset'))
test_dataset = load_from_disk(os.path.join(results_dir, 'test_dataset'))
val_dataset = load_from_disk(os.path.join(results_dir, 'val_dataset'))

# Fonction pour afficher un échantillon d'images
def show_sample(dataset, sample_size=20):
    sample = dataset.shuffle(seed=42).select(range(sample_size))
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    for i, (image, label) in enumerate(zip(sample['image'], sample['label'])):
        ax = axes[i // 5, i % 5]
        ax.imshow(image)
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Afficher un échantillon de 20 images pour chaque dataset
print("Train Dataset Sample:")
show_sample(train_dataset)

print("Test Dataset Sample:")
show_sample(test_dataset)

print("Validation Dataset Sample:")
show_sample(val_dataset)
