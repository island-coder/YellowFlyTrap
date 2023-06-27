import cv2
import matplotlib.pyplot as plt
import numpy as np

import preprocessing as pre
import hsv_seperation as hsv
import utility as util

img_wf = cv2.imread('assets/wf_test.jpg')
img_wf =cv2.medianBlur(img_wf,7)
img_wf_he=pre.hist_eq(img_wf)
img_wf_he=img_wf
img_wf_mask= hsv.hsv_sep_wf(img_wf_he)

plt.subplot(131)
plt.axis('off')
plt.title('whitefly')
plt.imshow(cv2.cvtColor(img_wf, cv2.COLOR_BGR2RGB)[1500:1900,750:1400,:])
plt.subplot(132)
plt.axis('off')
plt.title('whitefly preprocessed')
plt.imshow(cv2.cvtColor(img_wf_he, cv2.COLOR_BGR2RGB)[1500:1900,750:1400,:])
plt.subplot(133)
plt.axis('off')
plt.title('whitefly masked by HSV')
plt.imshow(img_wf_mask[1500:1900,750:1400,:])
plt.show()

img_macro=cv2.imread('assets/macro_test.jpg')
# util.viewBGR(img_macro)
img_macro=pre.extract_trap(img_macro)
img_macro =cv2.medianBlur(img_macro,7)
img_macro_he=pre.hist_eq(img_macro)
img_macro_mask= hsv.hsv_sep_macro(img_macro_he)

plt.subplot(131)
plt.axis('off')
plt.title('macro')
plt.imshow(cv2.cvtColor(img_macro, cv2.COLOR_BGR2RGB)[400:1800,70:824,:])
plt.subplot(132)
plt.axis('off')
plt.title('macro preprocessed')
plt.imshow(cv2.cvtColor(img_macro_he, cv2.COLOR_BGR2RGB)[400:1800,70:824,:])
plt.subplot(133)
plt.axis('off')
plt.title('macro masked by HSV')
plt.imshow(img_macro_mask[400:1800,70:824,:])
plt.show()

img_nesi=cv2.imread('assets/nesi_test.jpg')
# util.viewBGR(img_nesi)
img_nesi =cv2.medianBlur(img_nesi,7)
img_nesi_he=pre.hist_eq(img_nesi)
img_nesi_mask= hsv.hsv_sep_nesi(img_nesi_he)

plt.subplot(131)
plt.axis('off')
plt.title('nesi')
plt.imshow(cv2.cvtColor(img_nesi, cv2.COLOR_BGR2RGB)[500:2300,850:2700,:])
plt.subplot(132)
plt.axis('off')
plt.title('nesi preprocessed')
plt.imshow(cv2.cvtColor(img_nesi_he, cv2.COLOR_BGR2RGB)[500:2300,850:2700,:])
plt.subplot(133)
plt.axis('off')
plt.title('nesi masked by HSV')
plt.imshow(img_nesi_mask[500:2300,850:2700,:])
plt.show()