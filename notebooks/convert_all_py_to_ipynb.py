import os
import nbformat as nbf

# Chemin vers le répertoire contenant les fichiers .py
directory = '/home/xavier/code/truongxavier/datascientest/sprint_07'

# Parcourir tous les fichiers dans le répertoire et ses sous-répertoires
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.py'):
            # Chemin complet du fichier .py
            filepath = os.path.join(root, filename)

            # Lire le contenu du fichier .py
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            # Créer un nouveau notebook
            nb = nbf.v4.new_notebook()

            # Ajouter le code comme une nouvelle cellule dans le notebook
            nb.cells.append(nbf.v4.new_code_cell(code))

            # Chemin complet du fichier .ipynb
            ipynb_filename = os.path.splitext(filepath)[0] + '.ipynb'

            # Écrire le notebook dans un fichier .ipynb
            with open(ipynb_filename, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)

print("Conversion terminée !")
