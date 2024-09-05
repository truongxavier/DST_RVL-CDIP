import os
import nbformat
from nbconvert import PythonExporter

# Chemin vers le répertoire contenant les fichiers .ipynb
directory = '/home/xavier/code/truongxavier/datascientest/sprint_09/series_temporelles/exam'

# Parcourir tous les fichiers dans le répertoire
for filename in os.listdir(directory):
    if filename.endswith('.ipynb'):
        # Chemin complet du fichier .ipynb
        filepath = os.path.join(directory, filename)

        # Lire le fichier .ipynb
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)

        # Convertir en script Python
        exporter = PythonExporter()
        script, _ = exporter.from_notebook_node(notebook)

        # Chemin complet du fichier .py
        script_filename = os.path.splitext(filepath)[0] + '.py'

        # Écrire le script Python dans un fichier .py
        with open(script_filename, 'w', encoding='utf-8') as f:
            f.write(script)

print("Conversion terminée !")
