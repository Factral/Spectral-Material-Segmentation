import cv2
import glob
import os

# Configura el tamaño deseado para las imágenes redimensionadas
desired_size = (512, 512)

# Crea las nuevas carpetas si no existen
os.makedirs('DMS_v1/images_resized', exist_ok=True)
os.makedirs('DMS_v1/labels_resized', exist_ok=True)

# Redimensiona imágenes y máscaras
for image_path in glob.glob('DMS_v1/images/train/*.jpg'):
    base_name = os.path.basename(image_path)

    # Redimensiona la imagen RGB con interpolación bilineal
    print(image_path)
    img = cv2.imread(image_path)

    resized_img = cv2.resize(img, desired_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'DMS_v1/images_resized/{base_name}', resized_img)

    # Redimensiona la máscara correspondiente con interpolación más cercana
    label_path = image_path.replace('images', 'labels')
    label_path = label_path.replace('jpg', 'png')
    print(label_path)

    label = cv2.imread(label_path)
    resized_label = cv2.resize(label, desired_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f'DMS_v1/labels_resized/{base_name.replace("jpg", "png")}', resized_label)
