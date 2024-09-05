import pandas as pd
import os

def reduire_csv(chemin_acces):
    # Parcourir tous les fichiers dans le répertoire
    for fichier_csv in os.listdir(chemin_acces):
        # Vérifier si le fichier est un fichier CSV
        if fichier_csv.endswith('.csv'):
            # Combiner le chemin d'accès et le nom du fichier pour obtenir le chemin complet
            chemin_complet = os.path.join(chemin_acces, fichier_csv)

            # Lire le fichier CSV
            df = pd.read_csv(chemin_complet)

            # Garder l'en-tête et les 20 premières lignes
            df_reduit = df.head(20)

            # Extraire le nom de fichier et l'extension
            nom_fichier, extension = os.path.splitext(fichier_csv)

            # Créer le nouveau nom de fichier
            nouveau_nom_fichier = f"{nom_fichier}_petit{extension}"

            # Créer le chemin complet pour le nouveau fichier
            chemin_nouveau_fichier = os.path.join(chemin_acces, nouveau_nom_fichier)

            # Enregistrer le nouveau fichier CSV
            df_reduit.to_csv(chemin_nouveau_fichier, index=False)
            print(f"Fichier {nouveau_nom_fichier} créé.")

# Remplacez par le chemin d'accès à votre répertoire contenant les fichiers CSV
chemin_acces = '/home/xavier/code/truongxavier/datascientest/sprint_09/series_temporelles/exam'
reduire_csv(chemin_acces)
