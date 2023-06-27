import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_trap(img):
    img_gs=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.subplot(131)
    # plt.imshow(img_gs,'gray')
    img_gs=cv2.medianBlur(img_gs,15)
    # plt.subplot(132)
    # plt.imshow(img_gs,'gray')
    ret,thresh1 = cv2.threshold(img_gs,127,255,cv2.THRESH_BINARY)
    # plt.subplot(133)
    # plt.imshow(thresh1,'gray')
    # plt.show()
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    maxContour = max(contours, key = cv2.contourArea)
    # cv2.drawContours(img, maxContour, -1, (0, 255, 0), 3)
    x,y,w,h = cv2.boundingRect(maxContour)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[y:y+h,x:x+w,:])
    # plt.show()
    return img[y:y+h,x:x+w,:]

def hist_eq(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(img_hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(100,100))
    v_eq=clahe.apply(v)
    img_he=cv2.merge((h,s,v_eq))
    # plt.subplot(121)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.subplot(122)
    img_he=cv2.cvtColor(img_he,cv2.COLOR_HSV2BGR)
    # plt.imshow(img_he)
    # plt.show()
    return img_he
  
# img=cv2.imread('assets/macro_test.jpg')
# # plt.subplot(131)
# # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# img_blur=cv2.medianBlur(img,7)
# # plt.subplot(132)
# # plt.imshow(img_blur)

# img_he=hist_eq(img_blur)
# plt.subplot(133)
# plt.imshow(img_he)
# plt.show()
# print(img.shape,img_he.shape)

# hsv_gui.hsv_finder(img_he)


# img=cv2.imread('assets/mix.jpg')
# extract_trap(img)