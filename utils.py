#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''

# Necessary packages
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import shutil
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from IPython.core.display import display, HTML
import random
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
from random import randint
from sklearn.utils import shuffle

def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size "size"
    array = np.expand_dims(array, axis=0)
    return array

def print_in_color(txt_msg,fore_tupple,back_tupple,):
    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple 
    #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    rf,gf,bf=fore_tupple
    rb,gb,bb=back_tupple
    msg='{0}' + txt_msg
    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m' 
    print(msg .format(mat), flush=True)
    print('\33[0m', flush=True) # returns default print color to back to black
    return

def display_eval_metrics(e_data):
    msg='Model Metrics after Training'
    print_in_color(msg, (255,255,0), (55,65,80))
    msg='{0:^24s}{1:^24s}'.format('Metric', 'Value')
    print_in_color(msg, (255,255,0), (55,65,80))
    for key,value in e_data.items():
        print (f'{key:^24s}{value:^24.5f}')
    acc=e_data['accuracy']* 100
    return acc

def array2percent(array):
  #print(min(array), max(array))
  array2 = array - min(array)
  array3 = array2/np.sum(array2)*100
  #array = array/np.sum(array)*100
  #print(array3)
  return array3


def get_classlabel(class_code):
    labels = {0:'healthy', 1:'doubtful', 2:'minimal', 3:'moderate', 4:'severe'}
    #labels = {0.:0, 1.:1, 2.:2, 3.:3, 4.:4}
    return labels[class_code]

def get_images(directory):
    Images = []
    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street
    label = 0
    for labels in os.listdir(directory): #Main Directory where each class label is present as folder name.
        if labels == '0' : #Folder contain Glacier Images get the '2' class label.
            label = 0 #'healthy'
        elif labels == '1':
            label = 1#'doubtful'
        elif labels == '2':
            label = 2#'minimal'
        elif labels == '3':
            label = 3#'moderate'
        elif labels == '4':
            label = 4#'severe'
        for image_file in os.listdir(directory+'/'+labels): #Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory+'/'+labels+r'/'+image_file) #Reading the image (OpenCV)
            #print(directory+'/'+labels+r'/'+image_file)
            #image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)
            #image = scalarX(image)
            Images.append(image)
            Labels.append(label)
    return shuffle(Images,Labels)#,random_state=817328462) #Shuffle the dataset you just prepared.

def get_images2(directory): # function for image detection
  Images = []
  path = directory#os.path.join(directory)
  for img in os.listdir(path):
    img_array = cv2.imread(path+r'/'+img)
    #print(path+r'/'+img)
    #img_array = cv2.resize(img_array, (224, 224))
    img_array = scalarX(img_array)
    Images.append(img_array)
    #print(np.array(Images).shape)
  return Images#, random_state=819873262)



def plotImages2(link, Name):
    multipleImages = glob.glob(link)
    r = random.sample(multipleImages, 5)
    #print(r)
    #r2= r[0][32:]
    #r2= r[0].split("/")[-1][:-4]
    #print(r2)
    #fig = plt.figure(figsize=(20,20))
    fig = plt.figure(figsize=(20, 5))
    title_color = 'green' #if pred_grade == i else 'red'
    fig.color = 'green'
    fig.suptitle("Sample {} Xray Images".format(Name), fontsize=19,color='green')
    plt.subplot(151)
    plt.imshow(cv2.imread(r[0])); plt.axis('off')
    plt.title(r[0].split("/")[-1][:-4],color='r')
    plt.subplot(152)
    plt.imshow(cv2.imread(r[1])); plt.axis('off')
    plt.title(r[1].split("/")[-1][:-4],color='r')
    plt.subplot(153)
    plt.imshow(cv2.imread(r[2])); plt.axis('off')
    plt.title(r[2].split("/")[-1][:-4],color='r')
    plt.subplot(154)
    plt.imshow(cv2.imread(r[3])); plt.axis('off')
    plt.title(r[3].split("/")[-1][:-4],color='r')
    plt.subplot(155)
    plt.imshow(cv2.imread(r[4])); plt.axis('off')
    plt.title(r[4].split("/")[-1][:-4],color='r')
    


def data2df(link):
  image_dir = Path(link)
  # Get filepaths and labels
  filepaths = list(image_dir.glob(r'**/*.png'))
  labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
  filepaths = pd.Series(filepaths, name='Filepath').astype(str)
  labels = pd.Series(labels, name='Label_S')
  # Concatenate filepaths and labels
  image_df = pd.concat([filepaths, labels], axis=1)
  dict1={ '0':'healthy', '1':'doubtful', '2':'minimal', '3':'moderate', '4':'severe'}
  image_df['Label'] = image_df['Label_S'].map(dict1)
  return image_df

def check_data_folder(link):
  train_dir= link
  classes=sorted(os.listdir(train_dir))
  #print(classes)
  class_count=len(classes)
  for klass in classes:
      classpath=os.path.join(train_dir, klass)
      file_count=len(os.listdir(classpath))
      print ('Class: ' , klass, '   samples: ', file_count)

def balancing_class(link, max):
  # Cân bằng các lớp
  filepaths=[]
  labels=[]
  max_files=max # set maximum number of files in a class
  data_dir=r'/content/tommy/data'
  set_list=['train', 'test', 'val']
  class_totals=[0,0,0,0,0]
  for s in set_list:
      spath=os.path.join(data_dir,s)
      classes=sorted(os.listdir(spath)) # list of classes    
      for i, klass in enumerate(classes):        
          classpath=os.path.join(spath, klass)
          filelist=os.listdir(classpath)
          for f in filelist:
              fpath=os.path.join(classpath,f)
              # checks if the file extension is one that can be used with ImageDataGenerator
              index=fpath.rfind('.')
              ext=fpath[index+1:].lower()
              if ext in ['jpg', 'jpeg', 'png']:
                  # check to see if the image files are valid images
                  try:
                      img = Image.open(fpath)
                      img.verify()
                      if class_totals[i] < max_files:                                              
                          class_totals[i]=class_totals[i] + 1
                          filepaths.append(fpath)
                          labels.append(klass)                   
                  except:
                      print ('file ', fpath, ' is an invalid image file and will not be processed')             
              else:
                  print ('file ', fpath, ' with ', ext, '  will not be processed')       
  # determine the class weight dictionary
  for i in range(len(class_totals)):
      print (' Class: ', classes[i],  ' Totals_samples: ', class_totals[i])
  max_samples=np.max(class_totals)
  class_weight={}
  for i in range (len(class_totals)):
      class_weight[i]=max_samples/class_totals[i]
  print ('Class weight: ',class_weight)
  print(' Total files: ',len(filepaths), '  Total labels', len(labels))








