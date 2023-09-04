import cv2
import matplotlib.pyplot as plt
import numpy as np
import utility as util

def hsv_sep(img,low,high):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    hsv_mask=cv2.inRange(img_hsv,low,high)   
    img_masked=cv2.bitwise_and(img,img,mask=hsv_mask)
    # plt.imshow(img_masked)
    # plt.show()
    return img_masked,hsv_mask
    
# def hsv_sep_wf(img):
#     return hsv_sep(img,(0,0,210),(179,119,255))

def hsv_sep_wf(img):
    return hsv_sep(img,(0,0,0),(179,100,255))

# def hsv_sep_macro(img):
#     return hsv_sep(img,(0,0,0),(27,255,209))    

def hsv_sep_macro(img):
    return hsv_sep(img,(0,138,0),(179,243,255))    
       
# def hsv_sep_nesi(img):  #ori
#     return hsv_sep(img,(0,0,0),(24,250,212))

# def hsv_sep_nesi(img):
#     return hsv_sep(img,(25,0,0),(31,215,200))

def hsv_sep_nesi(img):
    return hsv_sep(img,(16,0,0),(33,200,200))
    
def hsv_sep_unkown(img):
    return hsv_sep(img,(29,0,0),(179,255,255))

