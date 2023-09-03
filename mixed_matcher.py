import cv2
import matplotlib.pyplot as plt
import numpy as np

import preprocessing as pre
import hsv_seperation as hsv
import utility as util

img = cv2.imread('assets/var/1277.jpg')
img=pre.extract_trap(img)

artefact_mask=pre.remove_artefacts(img)
unknown_class_mask=pre.remove_unkown_class(img)

img_blur =cv2.medianBlur(img,7)
img_he=pre.hist_eq(img_blur)


img_wf_mask,mask= hsv.hsv_sep_wf(img)
mask_wo_artefacts=cv2.bitwise_and(mask,mask,mask = artefact_mask)

contours, hierarchy = cv2.findContours(mask_wo_artefacts, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
     if  cv2.contourArea(cnt) >100 and cv2.contourArea(cnt) <1000:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[1500:1900,750:1400,:])
# plt.axis('off')
# plt.title('Whiteflies')
# plt.show()

img_macro_mask,mask= hsv.hsv_sep_macro(img_he)
mask_wo_unkown =cv2.bitwise_and(mask,mask,mask = unknown_class_mask)

mask_wo_artefacts=cv2.bitwise_and(mask_wo_unkown,mask_wo_unkown,mask = artefact_mask)
plt.subplot(131)
plt.title('artefact mask')
plt.imshow(artefact_mask,'gray')

plt.subplot(132)
plt.title('macro mask')
plt.imshow(mask,'gray')


plt.subplot(133)
plt.imshow(mask_wo_artefacts,'gray')
plt.show()

contours, hierarchy = cv2.findContours(mask_wo_artefacts, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
     if cv2.contourArea(cnt) >1500 and cv2.contourArea(cnt) <5000 and cv2.arcLength(cnt,True)<1000:
          # print(cv2.arcLength(cnt,True))
          x,y,w,h = cv2.boundingRect(cnt)
          cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[400:1800,70:824,:])
# plt.axis('off')
# plt.title('Macrolophus')
# plt.show()

img_nesi_mask,mask= hsv.hsv_sep_nesi(img_he)
mask_wo_artefacts=cv2.bitwise_and(mask,mask,mask = artefact_mask)
# plt.imshow(mask,'gray')
# plt.title('nesi mask')
# plt.axis('off')
# plt.show()

contours, hierarchy = cv2.findContours(mask_wo_artefacts, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
     if cv2.contourArea(cnt) >1500 and cv2.contourArea(cnt) <5000:
          x,y,w,h = cv2.boundingRect(cnt)
          cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
# plt.title('Nesidiocoris')
plt.show()