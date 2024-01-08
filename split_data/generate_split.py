import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import cv2
import glob

def main(args):
    # Directorio donde están tus imágenes y máscaras
    if args['dataset'] == 'dms':
        base_dir = '../dms_dataset/DMS_v1'
        images_dir = os.path.join(base_dir, 'images_resized')

        print(images_dir)

        filenames = glob.glob(images_dir + '/*.jpg')
        print(len(filenames))
        filenames = [os.path.basename(f) for f in filenames]

        train_files, test_files = train_test_split(filenames, test_size=0.5, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

        np.save(args['output_folder'] + '/train_files.npy', train_files)
        np.save(args['output_folder'] + '/test_files.npy', test_files)
        np.save(args['output_folder'] + '/val_files.npy', val_files)

        print(f"Train files: {len(train_files)}, Test files: {len(test_files)}, Val files: {len(val_files)}")


    elif args['dataset'] == 'localmat':

        base_dir = 'matbase/'
        images_dir = os.path.join(base_dir, 'images_resized')
        masks_dir = os.path.join(base_dir, 'masks_png')

        file_names = []

        for filename in os.listdir(images_dir):
            if filename.endswith(".png"): 
                file_names.append(filename)

        train_files, test_files = train_test_split(file_names, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)

        np.save(args['output_folder'] + '/train_files.npy', train_files)
        np.save(args['output_folder'] + '/test_files.npy', test_files)
        np.save(args['output_folder'] + '/val_files.npy', val_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, required=True, help='which dataset to split'
    )
    parser.add_argument(
        '--output_folder', type=str, default='.', help='where to save split'
    )
    args = parser.parse_args()
    main(vars(args))

