import cv2
import matplotlib.pyplot as plt
import numpy as np

import preprocessing as pre
import hsv_seperation as hsv
import utility as util


img_macro=cv2.imread('assets/macro_test.jpg',0)

img_macro_test=np.copy(img_macro)
img_macro_test[750:1000,970:1150]=0
macro_ref=img_macro[750:1000,970:1150]

util.viewBGR(img_macro)
util.viewBGR(img_macro_test)
util.viewBGR(macro_ref)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(macro_ref,None) 
kp2, des2 = orb.detectAndCompute(img_macro_test,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(macro_ref,kp1,img_macro_test,kp2,matches[:],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()