import sys
import os
import io
import pandas as pd
import shutil
import signal
import collections as col
import cv2 as cv
import argparse

from shutil import copyfile
from natsort import natsorted
from lxml import etree
from PIL import Image
from collections import namedtuple, OrderedDict


# Construct XML File for each image using information from CSV
def makeXMLFile(image, csv_file, csv_object_locs, train_image_dir, test_image_dir, city_name):
  
  annotation = etree.Element('annotation')

  fo = etree.Element('folder')
  fo.text = '/images/all/'

  annotation.append(fo)

  f = etree.Element('filename')
  f.text = image

  annotation.append(f)
  
  width = csv_file.iloc[csv_object_locs[0]]['width']
  height = csv_file.iloc[csv_object_locs[0]]['height']
  train_test = csv_file.iloc[csv_object_locs[0]]['train_test']

  size = etree.Element('size')
  w = etree.Element('width')
  w.text = str(width)
  h = etree.Element('height')
  h.text = str(height)
  d = etree.Element('depth')
  d.text = str(1)

  size.append(w)
  size.append(h)
  size.append(d)

  annotation.append(size)

  seg = etree.Element('segmented')
  seg.text = str(0)

  annotation.append(seg)
  
  # We create an object tag for each object in a photo and then append that to the annotation.  
  for dex, loc in enumerate(csv_object_locs):
    x_min = csv_file.iloc[loc]['xmin']
    y_min = csv_file.iloc[loc]['ymin']
    x_max = csv_file.iloc[loc]['xmax']
    y_max = csv_file.iloc[loc]['ymax']
    class_name = csv_file.iloc[loc]['class']
  
    if class_name == "TT":
      class_num = 0
    elif class_name == "DT":
      class_num = 1
    
    object = etree.Element('object')
    n = etree.Element('name')
    p = etree.Element('pose')
    t = etree.Element('truncated')
    d_1 = etree.Element('difficult')
    bb = etree.Element('bndbox')

    n.text = class_name
    p.text = 'center'
    t.text = str(1)
    d_1.text = str(0)

    xmi = etree.Element('xmin')
    ymi = etree.Element('ymin')
    xma = etree.Element('xmax')
    yma = etree.Element('ymax')

    xmi.text = str(x_min)
    yma.text = str(y_max)
    ymi.text = str(y_min)
    xma.text = str(x_max)

    bb.append(xmi)
    bb.append(ymi)
    bb.append(xma)
    bb.append(yma)

    object.append(n)
    object.append(p)
    object.append(t)
    object.append(d_1)
    object.append(bb)

    annotation.append(object)

  base_dir = os.path.join(r'/home/lab/Documents/bohao/data/transmission_line', 'yolo')

  if train_test == "train":
    xml_save_path = os.path.join(base_dir, "labels/train_{}/".format(city_name) + image[:-4] + ".xml")
    copyfile(os.path.join(base_dir, "images/all/" + image), train_image_dir + image)
  else:
    xml_save_path = os.path.join(base_dir, "labels/test_{}/".format(city_name) + image[:-4] + ".xml")
    copyfile(os.path.join(base_dir, "images/all/" + image), test_image_dir + image)
  
  print("Saving xml annotation for " + str(image) + " at " + str(xml_save_path))
  
  with open(xml_save_path, "wb") as file:
        file.write(etree.tostring(annotation, pretty_print=True))

if __name__ == '__main__':
  base_dir = os.path.join(r'/home/lab/Documents/bohao/data/transmission_line', 'yolo')
  city_name = 'Tucson'
  
  # Directory to all imagery.
  image_dir = os.path.join(base_dir, 'images/all/')
  
  # Directory to where training images are stored.
  train_image_dir = os.path.join(base_dir, "images/train_{}/".format(city_name))
  
  # Directory to where test images are stored.
  test_image_dir = os.path.join(base_dir, "images/test_{}/".format(city_name))
  
  # Directory to where labels (in XML) will be stored.
  labels_dir = os.path.join(base_dir, "csv")
  
  # Path to csv file from which information about training is pulled from.
  csv_file = pd.read_csv(os.path.join(labels_dir, "labels_{}.csv".format(city_name)))
  
  
  images = os.listdir(image_dir)
  images = natsorted([img for img in images if city_name in img])

  filenames = csv_file['filenames'].tolist()
  filenames = [filename[:-4] + ".png" for filename in filenames]
  
  # Iterate over each image present in the image_dir and generate an XML label if it contains a positive example.
  for index, image in enumerate(images):
    # If the image contains an annotation in the csv file, generate an XML.
    if image in filenames:
      csv_object_locs = csv_file.index[csv_file['filenames'] == image[:-4] + ".png"].tolist()
        
      makeXMLFile(image, csv_file, csv_object_locs, train_image_dir, test_image_dir, city_name)

