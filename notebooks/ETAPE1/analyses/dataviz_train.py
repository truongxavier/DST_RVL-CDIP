import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Chemin du fichier final
chemin_results = '/mnt/d/DVPT/projet/results/'
csv_final = 'final_processed_data_train.csv'

# Lire le fichier CSV final
df = pd.read_csv(chemin_results + csv_final)

# Créer un dossier pour enregistrer les graphiques
if not os.path.exists(chemin_results + 'visualizations/'):
    os.makedirs(chemin_results + 'visualizations/')

# 1. Boxplots pour détecter les outliers
plt.figure(figsize=(15, 10))

# Boxplot de la largeur des images
plt.subplot(2, 3, 1)
sns.boxplot(x=df['width'])
plt.title('Boxplot de la largeur des images')
plt.xlabel('Largeur')

# Boxplot de la hauteur des images
plt.subplot(2, 3, 2)
sns.boxplot(x=df['height'])
plt.title('Boxplot de la hauteur des images')
plt.xlabel('Hauteur')

# Boxplot de la luminosité moyenne
plt.subplot(2, 3, 3)
sns.boxplot(x=df['brightness'])
plt.title('Boxplot de la luminosité moyenne')
plt.xlabel('Luminosité moyenne')

# Boxplot du contraste
plt.subplot(2, 3, 4)
sns.boxplot(x=df['contrast'])
plt.title('Boxplot du contraste')
plt.xlabel('Contraste')

# Boxplot de l'entropie
plt.subplot(2, 3, 5)
sns.boxplot(x=df['entropy'])
plt.title('Boxplot de l\'entropie')
plt.xlabel('Entropie')

plt.tight_layout()
plt.savefig(chemin_results + 'visualizations/boxplots_data_quality_properties.png')

# 2. Heatmap de corrélation
plt.figure(figsize=(10, 8))

# Sélectionner uniquement les colonnes numériques
numeric_df = df.select_dtypes(include=[float, int])

correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap de corrélation')
plt.savefig(chemin_results + 'visualizations/correlation_heatmap.png')

# 3. Pairplot pour visualiser les relations entre les variables
sns.pairplot(numeric_df)
plt.savefig(chemin_results + 'visualizations/pairplot_data_quality_properties.png')

# 4. Histogrammes empilés pour des catégories spécifiques
# Exemple avec une colonne 'category' dans le DataFrame
if 'category' in df.columns:
    plt.figure(figsize=(15, 10))

    # Histogramme empilé de la largeur des images par catégorie
    plt.subplot(2, 3, 1)
    sns.histplot(data=df, x='width', hue='category', multiple='stack', bins=30)
    plt.title('Distribution de la largeur des images par catégorie')
    plt.xlabel('Largeur')
    plt.ylabel('Fréquence')

    # Histogramme empilé de la hauteur des images par catégorie
    plt.subplot(2, 3, 2)
    sns.histplot(data=df, x='height', hue='category', multiple='stack', bins=30)
    plt.title('Distribution de la hauteur des images par catégorie')
    plt.xlabel('Hauteur')
    plt.ylabel('Fréquence')

    # Histogramme empilé de la luminosité moyenne par catégorie
    plt.subplot(2, 3, 3)
    sns.histplot(data=df, x='brightness', hue='category', multiple='stack', bins=30)
    plt.title('Distribution de la luminosité moyenne par catégorie')
    plt.xlabel('Luminosité moyenne')
    plt.ylabel('Fréquence')

    # Histogramme empilé du contraste par catégorie
    plt.subplot(2, 3, 4)
    sns.histplot(data=df, x='contrast', hue='category', multiple='stack', bins=30)
    plt.title('Distribution du contraste par catégorie')
    plt.xlabel('Contraste')
    plt.ylabel('Fréquence')

    # Histogramme empilé de l'entropie par catégorie
    plt.subplot(2, 3, 5)
    sns.histplot(data=df, x='entropy', hue='category', multiple='stack', bins=30)
    plt.title('Distribution de l\'entropie par catégorie')
    plt.xlabel('Entropie')
    plt.ylabel('Fréquence')

    plt.tight_layout()
    plt.savefig(chemin_results + 'visualizations/stacked_histograms_by_category.png')

# 5. Analyse des statistiques descriptives
stats = df.describe()
print(stats)
stats.to_csv(chemin_results + 'visualizations/descriptive_statistics.csv')

# 6. Graphiques de densité
plt.figure(figsize=(15, 10))

# Densité de la largeur des images
plt.subplot(2, 3, 1)
sns.kdeplot(df['width'], shade=True)
plt.title('Densité de la largeur des images')
plt.xlabel('Largeur')
plt.ylabel('Densité')

# Densité de la hauteur des images
plt.subplot(2, 3, 2)
sns.kdeplot(df['height'], shade=True)
plt.title('Densité de la hauteur des images')
plt.xlabel('Hauteur')
plt.ylabel('Densité')

# Densité de la luminosité moyenne
plt.subplot(2, 3, 3)
sns.kdeplot(df['brightness'], shade=True)
plt.title('Densité de la luminosité moyenne')
plt.xlabel('Luminosité moyenne')
plt.ylabel('Densité')

# Densité du contraste
plt.subplot(2, 3, 4)
sns.kdeplot(df['contrast'], shade=True)
plt.title('Densité du contraste')
plt.xlabel('Contraste')
plt.ylabel('Densité')

# Densité de l'entropie
plt.subplot(2, 3, 5)
sns.kdeplot(df['entropy'], shade=True)
plt.title('Densité de l\'entropie')
plt.xlabel('Entropie')
plt.ylabel('Densité')

# Densité de la variance locale moyenne (si elle existe)
if 'mean_local_var' in df.columns:
    plt.subplot(2, 3, 6)
    sns.kdeplot(df['mean_local_var'], shade=True)
    plt.title('Densité de la variance locale moyenne')
    plt.xlabel('Variance locale moyenne')
    plt.ylabel('Densité')
else:
    print("La colonne 'mean_local_var' n'existe pas dans le DataFrame.")

plt.tight_layout()
plt.savefig(chemin_results + 'visualizations/density_plots.png')
