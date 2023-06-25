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