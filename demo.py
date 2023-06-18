import cv2
import matplotlib.pyplot as plt
import numpy as np

img_macro=cv2.imread('assets/macro_test.jpg')
img_macro=cv2.cvtColor(img_macro,cv2.COLOR_BGR2RGB)
img_macro=img_macro[350:1900,450:1200,:]
plt.imshow(img_macro)
# plt.show()

# (hMin = 19 , sMin = 0, vMin = 160), (hMax = 179 , sMax = 238, vMax = 254)

img_hsv=cv2.cvtColor(img_macro,cv2.COLOR_RGB2HSV)
hsv_mask=cv2.inRange(img_hsv,(0,19,160),(179,238,254))

plt.imshow(hsv_mask,'gray')
# plt.show()

img_init_masked=cv2.bitwise_and(img_macro,img_macro,mask=hsv_mask) #inital mask
plt.imshow(img_init_masked)
# plt.show()

img_nesi=cv2.imread('assets/nesi_test.jpg')
img_nesi=cv2.cvtColor(img_nesi,cv2.COLOR_BGR2RGB)
img_nesi=img_nesi[550:2400,2000:2600,:]
# plt.imshow(img_nesi)
# plt.show()

img_hsv=cv2.cvtColor(img_nesi,cv2.COLOR_RGB2HSV)
hsv_mask=cv2.inRange(img_hsv,(0,0,0),(179,250,255))
# plt.imshow(hsv_mask,'gray')
# plt.show()

img_init_masked=cv2.bitwise_and(img_nesi,img_nesi,mask=hsv_mask) #inital mask
plt.imshow(img_init_masked)
# plt.show()


img_wf=cv2.imread('assets/wf_test.jpg')
img_wf=cv2.cvtColor(img_wf,cv2.COLOR_BGR2RGB)
img_wf=img_wf[:,:,:]
plt.imshow(img_wf)
plt.show()

# (hMin = 0 , sMin = 0, vMin = 210), (hMax = 179 , sMax = 172, vMax = 255) wf

img_hsv=cv2.cvtColor(img_wf,cv2.COLOR_RGB2HSV)
hsv_mask=cv2.inRange(img_hsv,(0,0,210),(179,172,255))
plt.imshow(hsv_mask,'gray')
plt.show()

img_init_masked=cv2.bitwise_and(img_wf,img_wf,mask=hsv_mask) #inital mask
plt.imshow(img_init_masked)
plt.show()
