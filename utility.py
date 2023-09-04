import cv2
import matplotlib.pyplot as plt
import numpy as np

def viewBGR(img): #for BGR
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def viewRGB(img): #for BGR
    plt.imshow(img)
    plt.show()

import xml.etree.ElementTree as ET

def count_insect_classes(filename):
  """Counts the number of insect classes in an XML file.

  Args:
    filename: The path to the XML file.

  Returns:
    A dictionary mapping insect class names to the number of times they appear in the file.
  """
  tree = ET.parse(filename)
  root = tree.getroot()

  insect_classes = {}
  for object in root.findall('object'):
    insect_class = object.find('name').text
    if insect_class not in insect_classes:
      insect_classes[insect_class] = 0
    insect_classes[insect_class] += 1

  return insect_classes


# import os

# folder_path = 'E:/CS/300L/CS314/StickyTrapsOriginal/45d71d06-7fea-4f35-9ced-6e793017b254_1_all/4TUDatasetAnonymised'

# for filename in os.listdir(folder_path):
#     if filename.endswith('.xml'):
#         insect_classes = count_insect_classes(os.path.join(folder_path, filename))
#         if 'NC' in insect_classes:
#             print(f'{filename} counts:',end="_")
#             for insect_class, count in insect_classes.items():
#                 print(f'{insect_class}: {count}',end=", ")
#             print("")

# insect_classes = count_insect_classes('assets/var/1117.xml')

# for insect_class, count in insect_classes.items():
#     print(f'{insect_class}: {count}')
