import os
import numpy as np
from sklearn.model_selection import train_test_split

# Directorio donde están tus imágenes y máscaras
base_dir = 'matbase/'
images_dir = os.path.join(base_dir, 'images_resized')
masks_dir = os.path.join(base_dir, 'masks_png')

# Lista para almacenar los nombres de los archivos
file_names = []

# Asumiendo que cada imagen tiene una máscara correspondiente con el mismo nombre
for filename in os.listdir(images_dir):
    if filename.endswith(".png"):  # Asegúrate de que sean archivos PNG
        file_names.append(filename)

# Dividir en conjuntos de entrenamiento y prueba
train_files, test_files = train_test_split(file_names, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)

# Guardar los nombres de archivo en archivos .npy
np.save('train_files.npy', train_files)
np.save('test_files.npy', test_files)
np.save('val_files.npy', val_files)

print("División completada. Archivos guardados como 'train_files.npy' y 'test_files.npy'")
