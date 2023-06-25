import cv2
import matplotlib.pyplot as plt
import numpy as np

def hsv_sep(img,low,high):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    hsv_mask=cv2.inRange(img_hsv,low,high)   
    img_masked=cv2.bitwise_and(img,img,mask=hsv_mask)
    plt.imshow(img_masked)
    plt.show()
    
def hsv_sep_wf(img):
    hsv_sep(img,(0,0,210),(179,172,255))
    
def hsv_sep_nesi(img):
    hsv_sep(img,(0,0,0),(179,250,255))
    
def hsv_sep_macro(img):
    hsv_sep(img,(0,19,160),(179,238,254))    
    
img=cv2.imread('assets/wf_test.jpg')

hsv_sep_wf(img)


    