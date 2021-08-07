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
from IPython.display import display, Markdown
from glob2 import glob



def scaler(img): # normal preprocessing function used originally but poor accuracy on test set
    return img/127.5 -1 # scale the pixels between -1 and + 1
    
def scalar(img):# try using adaptive thresholding on images to improve training accuracy
    img=np.array(img, dtype='uint8')
    img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2);
    img=cv2.cvtColor(th2, cv2.COLOR_GRAY2RGB)
    img=np.array(img, dtype=('float32'))  
    img=img/127.5-1
    return img

def scalarX(img):# try using adaptive thresholding on images to improve training accuracy
    img=np.array(img, dtype='uint8')
    img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2);
    img=cv2.cvtColor(th2, cv2.COLOR_GRAY2RGB)
    img=np.array(img, dtype=('float32'))  
    img=img/127.5-1
    return img

#@title Function defines

#@title Pre-processing Methods
# Proposed fuzzy method
def Infer(i, M, get_fuzzy_set=False):
    # Calculate degree of membership for each class
    VD = VeryDark(i, M)
    Da = Dark(i, M)
    SD = SlightlyDark(i, M)
    SB = SlightlyBright(i, M)
    Br = Bright(i, M)
    VB = VeryBright(i, M)
    
    # Fuzzy Inference:
    x = np.arange(-50, 306)
    Inferences = (
        OutputFuzzySet(x, ExtremelyDark, M, VD),
        OutputFuzzySet(x, VeryDark, M, Da),
        OutputFuzzySet(x, Dark, M, SD),
        OutputFuzzySet(x, Bright, M, SB),
        OutputFuzzySet(x, VeryBright, M, Br),
        OutputFuzzySet(x, ExtremelyBright, M, VB)
    )
    
    # Calculate AggregatedFuzzySet:
    fuzzy_output = AggregateFuzzySets(Inferences)
    
    # Calculate crisp value of centroid
    if get_fuzzy_set:
        return np.average(x, weights=fuzzy_output), fuzzy_output
    return np.average(x, weights=fuzzy_output)


def VeryDark(x, M):
    return G(x, 0, M/6)

def OutputFuzzySet(x, f, M, thres):
    x = np.array(x)
    result = f(x, M)
    result[result > thres] = thres
    return result

def AggregateFuzzySets(fuzzy_sets):
    return np.max(np.stack(fuzzy_sets), axis=0)

def ExtremelyDark(x, M):
    return G(x, -50, M/6)
    
# Gaussian Function:
def G(x, mean, std):
    return np.exp(-0.5*np.square((x-mean)/std))

def Dark(x, M):
    return G(x, M/2, M/6)

def SlightlyDark(x, M):
    return G(x, 5*M/6, M/6)

def SlightlyBright(x, M):
    return G(x, M+(255-M)/6, (255-M)/6)

def Bright(x, M):
    return G(x, M+(255-M)/2, (255-M)/6)

def VeryBright(x, M):
    return G(x, 255, (255-M)/6)

def ExtremelyBright(x, M):
    return G(x, 305, (255-M)/6)

def FuzzyContrastEnhance(rgb):
    # Convert RGB to LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    
    # Get L channel
    l = lab[:, :, 0]
    
    # Calculate M value
    M = np.mean(l)
    if M < 128:
        M = 127 - (127 - M)/2
    else:
        M = 128 + M/2
        
    # Precompute the fuzzy transform
    x = list(range(-50,306))
    FuzzyTransform = dict(zip(x,[Infer(np.array([i]), M) for i in x]))
    
    # Apply the transform to l channel
    u, inv = np.unique(l, return_inverse = True)
    l = np.array([FuzzyTransform[i] for i in u])[inv].reshape(l.shape)
    
    # Min-max scale the output L channel to fit (0, 255):
    Min = np.min(l)
    Max = np.max(l)
    lab[:, :, 0] = (l - Min)/(Max - Min) * 255
    
    # Convert LAB to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Traditional method of histogram equalization
def HE(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Contrast Limited Adaptive Histogram Equalization
def CLAHE(rgb):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def CLAHE2(img):
    dataxxx = np.array(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
    clahe = CLAHE(dataxxx)
    return clahe

def CLAHE3(img):
  img=np.array(img, dtype='uint8')
  img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
  clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
  img = clahe.apply(img)
  img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  img=np.array(img, dtype=('float32'))  
  img=img/127.5-1
  return img
  

def show_anh_tienxuly(link):
    PATH = link#'/content/tommy/15.07.1112'
    data = np.array([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in glob(f'{PATH}/*')])
    data.shape
    for i in range(data.shape[0]):
        img = data[i]
        fce = FuzzyContrastEnhance(img)
        he = HE(img)
        clahe = CLAHE(img) 
        display(Markdown(f'### <p style="text-align: center;">Sample Photo {i+1} of Cô Lan Dataset</p>'))
        plt.figure(figsize=(20, 18))
        plt.subplot(1, 3, 1)
        plt.imshow(data[i])
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(fce)
        plt.title('Fuzzy Contrast Enhance')
        plt.axis('off')
        
        #plt.subplot(2, 2, 3)
        #plt.imshow(he)
        #plt.title('Traditional HE')
        #plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(clahe)
        plt.title('CLAHE')
        
        plt.axis('off')
        plt.show()


def preVert(image_path, output_home):
    i = 0#1
    limit = 2.0
        
    #image_path = os.path.join(image_dir, f)   
    img = cv2.imread(image_path)

    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    img_eq = cv2.equalizeHist(y)
    img_y_cr_cb_eq = cv2.merge((img_eq, cr, cb))
    img = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    clahe = CLAHE(img)
    clahe = CLAHE(clahe)
    clahe = CLAHE(clahe)

    output_dir = output_home # os.path.join(output_home, basename)
    cv2.imwrite(output_home, clahe)










