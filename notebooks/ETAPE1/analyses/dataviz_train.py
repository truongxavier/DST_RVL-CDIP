import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chemin du fichier final
chemin_results = '/mnt/d/DVPT/projet/results/'
csv_final = 'final_processed_data_train.csv'

# Lire le fichier CSV final
df = pd.read_csv(chemin_results + csv_final)

# Afficher les colonnes disponibles
print("Colonnes disponibles dans le DataFrame :")
print(df.columns)

# Configuration de base pour les graphiques
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))

# Histogramme de la largeur des images
plt.subplot(2, 3, 1)
sns.histplot(df['width'], kde=True, bins=30)
plt.title('Distribution de la largeur des images')
plt.xlabel('Largeur')
plt.ylabel('Fréquence')

# Histogramme de la hauteur des images
plt.subplot(2, 3, 2)
sns.histplot(df['height'], kde=True, bins=30)
plt.title('Distribution de la hauteur des images')
plt.xlabel('Hauteur')
plt.ylabel('Fréquence')

# Histogramme de la luminosité moyenne
plt.subplot(2, 3, 3)
sns.histplot(df['brightness'], kde=True, bins=30)
plt.title('Distribution de la luminosité moyenne')
plt.xlabel('Luminosité moyenne')
plt.ylabel('Fréquence')

# Histogramme du contraste
plt.subplot(2, 3, 4)
sns.histplot(df['contrast'], kde=True, bins=30)
plt.title('Distribution du contraste')
plt.xlabel('Contraste')
plt.ylabel('Fréquence')

# Histogramme de l'entropie
plt.subplot(2, 3, 5)
sns.histplot(df['entropy'], kde=True, bins=30)
plt.title('Distribution de l\'entropie')
plt.xlabel('Entropie')
plt.ylabel('Fréquence')

# Vérifier si la colonne 'mean_local_var' existe avant de tracer l'histogramme
if 'mean_local_var' in df.columns:
    # Histogramme de la variance locale moyenne
    plt.subplot(2, 3, 6)
    sns.histplot(df['mean_local_var'], kde=True, bins=30)
    plt.title('Distribution de la variance locale moyenne')
    plt.xlabel('Variance locale moyenne')
    plt.ylabel('Fréquence')
else:
    print("La colonne 'mean_local_var' n'existe pas dans le DataFrame.")

# Ajuster la mise en page
plt.tight_layout()
plt.savefig(chemin_results + 'data_quality_properties_visualization.png')
plt.show()

# Afficher les statistiques descriptives
print("Statistiques descriptives :")
print(df.describe())

# Filtrer les colonnes numériques pour la matrice de corrélation
numeric_df = df.select_dtypes(include=[float, int])

# Matrice de corrélation
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation des propriétés des images')
plt.savefig(chemin_results + 'correlation_matrix.png')
plt.show()
