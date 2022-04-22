#%%
import numpy as np
import pandas as pd
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

def load_bounding_box_annotations(dataset_path=''):
  
  bboxes = {}
  
  with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      bbox = map(int, pieces[1:])
      bboxes[image_id] = bbox
  
  return bboxes

def load_part_annotations(dataset_path=''):
  
  parts = {}
  
  with open(os.path.join(dataset_path, 'parts/part_locs.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      parts.setdefault(image_id, [0] * 11)
      part_id = int(pieces[1])
      parts[image_id][part_id] = map(int, pieces[2:])

  return parts  
  
def load_part_names(dataset_path=''):
  
  names = {}

  with open(os.path.join(dataset_path, 'parts/parts.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      part_id = int(pieces[0])
      names[part_id] = ' '.join(pieces[1:])
  
  return names  
    
def load_class_names(dataset_path=''):
  
  names = {}
  
  with open(os.path.join(dataset_path, 'classes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      class_id = pieces[0]
      names[class_id] = ' '.join(pieces[1:])
  
  return names

def load_image_labels(dataset_path=''):
  labels = {}
  
  with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      class_id = pieces[1]
      labels[image_id] = class_id
  
  return labels
        
def load_image_paths(dataset_path='', path_prefix=''):
  
  paths = {}
  
  with open(os.path.join(dataset_path, 'images.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      path = os.path.join(path_prefix, pieces[1])
      paths[image_id] = path
  
  return paths

def load_image_sizes(dataset_path=''):
  
  sizes = {}
  
  with open(os.path.join(dataset_path, 'sizes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      width, height = map(int, pieces[1:])
      sizes[image_id] = [width, height]
  
  return sizes

def load_hierarchy(dataset_path=''):
  
  parents = {}
  
  with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      child_id, parent_id = pieces
      parents[child_id] = parent_id
  
  return parents

def load_photographers(dataset_path=''):
  
  photographers = {}
  with open(os.path.join(dataset_path, 'photographers.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      photographers[image_id] = ' '.join(pieces[1:])
  
  return photographers

def load_train_test_split(dataset_path=''):
  train_images = []
  test_images = []
  
  with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      is_train = int(pieces[1])
      if is_train:
        train_images.append(image_id)
      else:
        test_images.append(image_id)
        
  return train_images, test_images 

#%%
base_path = 'E:/Datasets/NABirds/nabirds'
image_path  = 'images'

# Load in the image data
# Assumes that the images have been extracted into a directory called "images"
image_paths = load_image_paths(base_path, path_prefix=image_path)
image_sizes = load_image_sizes(base_path)
image_bboxes = load_bounding_box_annotations(base_path)
image_parts = load_part_annotations(base_path)
image_class_labels = load_image_labels(base_path)

# Load in the class data
class_names = load_class_names(base_path)
class_hierarchy = load_hierarchy(base_path)

BATCH_SIZE = 4
IMG_HEIGHT = 600
IMG_WIDTH = 600

#%%

image_names = pd.read_csv(base_path+"/images.txt",
                          sep=" ",
                          header=None,
                          names=["image_id", "image_name"],
                          index_col="image_id")

bounding_boxes = pd.read_csv(base_path+"/bounding_boxes.txt",
                             sep=" ",
                             header=None,
                             names=["image_id", "x", "y", "width", "height"],
                             index_col="image_id")

class_labels = pd.read_csv(base_path+"/image_class_labels.txt",
                           sep=" ",
                           header=None,
                           names=["image_id", "class_id"],
                           index_col="image_id")

image_sizes = pd.read_csv(base_path+"/sizes.txt",
                          sep=" ",
                          header=None,
                          names=["image_id", "iwidth", "iheight"],
                          index_col="image_id")

class_names = load_class_names(base_path)
class_names = pd.json_normalize(class_names).T.reset_index()
class_names.columns = ['class_id', 'class_name']
class_names.class_id = class_names.class_id.astype(int)


image_info = image_names.join(bounding_boxes)\
    .join(class_labels)\
    .join(image_sizes)\
    .merge(class_names, on="class_id")

dataset = image_info\
    .dropna()\
    .apply(lambda r: {
        "filename":os.path.join(base_path,'images', r["image_name"]),
        "id":r.name,
        "class": {
            "label":r["class_id"]+1,
            "text":r["class_name"].replace("'", "")
            },
        "object": {
            "id":["1"],
            "count": 1,
            "area":[ float(r["width"]*r["height"])/(r["iheight"]*r["iwidth"])],
            "bbox":{
                "xmin":[float(r["x"])/r["iwidth"]],
                "xmax":[float(r["x"]+r["width"])/r["iwidth"]],
                "ymin":[float(r["y"])/r["iheight"]],
                "ymax":[float(r["y"]+r["height"])/r["iheight"]],
                "label": [(1+r["class_id"])]
                } 
            }
        },axis=1)\
    .values\
    .tolist()

#%% Documentation for creating TF Record using the Object Detection API 
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-label-map