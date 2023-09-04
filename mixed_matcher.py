import cv2
import matplotlib.pyplot as plt
import numpy as np

import preprocessing as pre
import hsv_seperation as hsv
import utility as util

file=1113
img_path=f'assets/var/{file}.jpg'
annotation_path=f'assets/var/{file}.xml'

detected_classes={'WF':0,'MR':0,'NC':0}
actual_classes=util.count_insect_classes(annotation_path)

img = cv2.imread(img_path)
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
        detected_classes['WF']+=1
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[1500:1900,750:1400,:])
# plt.axis('off')
# plt.title('Whiteflies')
# plt.show()

img_macro_mask,mask= hsv.hsv_sep_macro(img_he)
mask_wo_unkown =cv2.bitwise_and(mask,mask,mask = unknown_class_mask)

mask_wo_artefacts=cv2.bitwise_and(mask_wo_unkown,mask_wo_unkown,mask = artefact_mask)
plt.subplot(231)
plt.title('artefact mask')
plt.imshow(artefact_mask,'gray')

plt.subplot(232)
plt.title('macro mask')
plt.imshow(mask,'gray')

plt.subplot(233)
plt.title('unknown mask')
plt.imshow(unknown_class_mask,'gray')

plt.subplot(234)
plt.title('macro wo unkown')
plt.imshow(mask_wo_unkown,'gray')

plt.subplot(235)
plt.title('mask_wo_artefacts')
plt.imshow(mask_wo_artefacts,'gray')
plt.show()

contours, hierarchy = cv2.findContours(mask_wo_artefacts, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
     if cv2.contourArea(cnt) >1500 and cv2.contourArea(cnt) <5000 and cv2.arcLength(cnt,True)<1000:
          # print(cv2.arcLength(cnt,True))
          x,y,w,h = cv2.boundingRect(cnt)
          aspect_ratio = float(w)/h
          if aspect_ratio>0.2 and aspect_ratio <1.5:
               cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
               detected_classes['MR']+=1
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[400:1800,70:824,:])
# plt.axis('off')
# plt.title('Macrolophus')
# plt.show()

img_nesi_mask,mask= hsv.hsv_sep_nesi(img_blur)
mask_wo_unkown =cv2.bitwise_and(mask,mask,mask = unknown_class_mask)
mask_wo_artefacts=cv2.bitwise_and(mask_wo_unkown,mask_wo_unkown,mask = artefact_mask)

kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))    
mask_wo_artefacts=cv2.morphologyEx(mask_wo_artefacts, cv2.MORPH_ERODE, kernel)
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(41,41))    
mask_wo_artefacts = cv2.morphologyEx(mask_wo_artefacts, cv2.MORPH_DILATE, kernel)

plt.subplot(231)
plt.title('artefact mask')
plt.imshow(artefact_mask,'gray')

plt.subplot(232)
plt.title('nesi mask')
plt.imshow(mask,'gray')

plt.subplot(233)
plt.title('unknown mask')
plt.imshow(unknown_class_mask,'gray')

plt.subplot(234)
plt.title('nesi wo unkown')
plt.imshow(mask_wo_unkown,'gray')

plt.subplot(235)
plt.title('nesi mask_wo_artefacts')
plt.imshow(mask_wo_artefacts,'gray')
plt.show()

contours, hierarchy = cv2.findContours(mask_wo_artefacts, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
     if cv2.contourArea(cnt) >2000 and cv2.contourArea(cnt) <5000:
          x,y,w,h = cv2.boundingRect(cnt)
          aspect_ratio = float(w)/h
          if aspect_ratio>0.2 and aspect_ratio <1.5:
               cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
               detected_classes['NC']+=1


font = cv2.FONT_HERSHEY_SIMPLEX
h,w=img.shape[0:2]
i=1
for insect_class, count in detected_classes.items():
#     print(f'{insect_class}: {count}')
     if count!=0:
          cv2.putText(img,f'{insect_class}: {count}',(30,h-150*i), font, 4,(128,0,128),6,cv2.LINE_AA)
          i+=1

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
# plt.title('Nesidiocoris')
plt.show()


print('***Detected***')
for insect_class, count in detected_classes.items():
    print(f'{insect_class}: {count}')
print('***Actual`***')
for insect_class, count in actual_classes.items():
    print(f'{insect_class}: {count}')
