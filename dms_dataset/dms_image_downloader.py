# python3
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Open Images image downloader for DMS dataset.

This script downloads a subset of Open Images images provided in the 
info.json.gz file of the DMS dataset. The images are first checked for 
on the AWS S3 bucket, and if not found, they are downloaded from the
original URL. The script can be run in parallel to speed up the download.

Based on code from: 
https://github.com/apple/ml-dms-dataset/blob/main/prepare_images.py
https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
"""

import argparse
from concurrent import futures
import os
import re
import sys
import json
import gzip
import urllib
import posixpath

import boto3
import botocore
import tqdm

BUCKET_NAME = 'open-images-dataset'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'


def check_and_homogenize_one_image(image):
  split, image_id = re.match(REGEX, image).groups()
  yield split, image_id


def check_and_homogenize_image_list(image_list):
  for line_number, image in enumerate(image_list):
    try:
      yield from check_and_homogenize_one_image(image)
    except (ValueError, AttributeError):
      raise ValueError(
          f'ERROR in line {line_number} of the image list. The following image '
          f'string is not recognized: "{image}".')


def read_image_list_file(image_list_file):
  with open(image_list_file, 'r') as f:
    for line in f:
      yield line.strip().replace('.jpg', '')


def download_one_image(bucket, img_path, url, split, image_id, download_folder, verbose):
  original_name = posixpath.split(urllib.parse.urlparse(url).path)[1]
  local_file_path = os.path.join(download_folder, original_name)
  if os.path.exists(local_file_path):
    # check if the image exists locally
    if verbose:
      print(f'Skipping image {img_path}, already exists.')
    return None
  try:
    bucket.download_file(f'{split}/{image_id}.jpg',
                         os.path.join(download_folder, original_name))
    return image_id
  except botocore.exceptions.ClientError as exception:
    try:
      # download directly from the original url using requests
      urllib.request.urlretrieve(url, os.path.join(download_folder, original_name))
      return image_id
    except Exception as exception:
      if verbose:
        print(f'Failed to download image {original_name}: {exception}')
      return exception, split + '/' + image_id, url



def download_all_images(args):
  """Downloads all images specified in the input file."""
  bucket = boto3.resource(
      's3', config=botocore.config.Config(
          signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)

  download_folder = args['download_folder'] or os.getcwd()
  verbose = args['verbose']

  data_path = args['data_path']
  data = json.loads(
      gzip.open(os.path.join(data_path, 'info.json.gz'), 'rb').read()
  )
  
  split = []
  image_id = []
  original_urls = []
  image_paths = []
  for datum in data:
      original_urls.append(datum['openimages_metadata']['OriginalURL'])
      split.append(datum['openimages_metadata']['Subset'])
      image_id.append(datum['openimages_metadata']['ImageID'])
      image_paths.append(datum['image_path'])

  if not os.path.exists(download_folder):
    os.makedirs(download_folder)

  downloaded_images = []
  errored_images = []
  errored_urls = []
  progress_bar = tqdm.tqdm(
      total=len(image_id), desc='Downloading images', leave=True)
  with futures.ThreadPoolExecutor(
      max_workers=args['num_processes']) as executor:
    all_futures = [
        executor.submit(download_one_image, bucket, img_path, url, split, image_id,
                        download_folder, verbose) for (img_path, url, split, image_id) in zip(image_paths, original_urls, split, image_id)
    ]
    for future in futures.as_completed(all_futures):
      result = future.result()
      if isinstance(result, str):
        downloaded_images.append(image_id)
        progress_bar.update(1)
      elif result is not None:
        errored_images.append(result[1])
        errored_urls.append(result[2])
        progress_bar.update(1)
      else:
        progress_bar.update(1)

  progress_bar.close()

  num_skipped_images = len(image_id) - len(downloaded_images) - len(
      errored_images)
  
  print(f'{num_skipped_images} images skipped (already downloaded).')
  print(f'{len(errored_images)} images errored.')
  print(f'{len(downloaded_images)} images downloaded.')

  # save the list of errored images
  if len(errored_images) > 0:
    # save in same folder as image_list which is a file that ends in .txt
    errored_images_file = args["download_folder"] + '/errored_images.txt'
    with open(errored_images_file, 'w') as f:
      for image_id in errored_images:
        f.write(f'{image_id}' + '\n')
    errored_urls_file = args["download_folder"] + '/errored_urls.txt'
    with open(errored_urls_file, 'w') as f:
      for image_url in errored_urls:
        f.write(f'{image_url}' + '\n')

    print(f'List of errored images saved to {errored_images_file}.')
    print(f'List of errored urls saved to {errored_urls_file}.')



if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      '--data_path',
      type=str,
      default='/DMS_v1',
      help=('Filename that contains the split + image IDs of the images to '
            'download. Check the document'))
  parser.add_argument(
      '--num_processes',
      type=int,
      default=5,
      help='Number of parallel processes to use (default is 5).')
  parser.add_argument(
      '--download_folder',
      type=str,
      default='/DMS_v1',
      help='Folder where to download the images.')
  parser.add_argument('--verbose', type=bool, default=False, help='Verbose output.')

  url = 'https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms_v1_labels.zip'
  filename = 'dms_v1_labels.zip'
  urllib.request.urlretrieve(url, filename)
  os.system('unzip dms_v1_labels.zip')


  download_all_images(vars(parser.parse_args()))
