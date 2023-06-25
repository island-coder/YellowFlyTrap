import cv2
import matplotlib.pyplot as plt
import numpy as np

import hsv_finder as hsv_gui

def hist_eq(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(img_hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(500,500))
    v_eq=clahe.apply(v)
    img_he=cv2.merge((h,s,v_eq))
    # plt.subplot(121)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.subplot(122)
    img_he=cv2.cvtColor(img_he,cv2.COLOR_HSV2RGB)
    plt.imshow(img_he)
    # plt.show()
    return img_he
  
img=cv2.imread('assets/wf_test.jpg')
plt.subplot(131)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

img_blur=cv2.medianBlur(img,7)
plt.subplot(132)
plt.imshow(img_blur)

img_he=hist_eq(img_blur)
plt.subplot(133)
plt.imshow(img_he)
plt.show()
print(img.shape,img_he.shape)
# hsv_gui.hsv_finder(cv2.cvtColor(img_he,cv2.COLOR_RGB2BGR))
