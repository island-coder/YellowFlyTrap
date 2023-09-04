import cv2
import matplotlib.pyplot as plt
import numpy as np
import hsv_seperation as hsv


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


def remove_artefacts(img):
    # img=hist_eq(img)
    img_gs=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.subplot(221)
    # plt.imshow(img_gs,'gray')
    img_gs=cv2.medianBlur(img_gs,15)
    # plt.subplot(222)
    # plt.imshow(img_gs,'gray')
    ret,thresh1 = cv2.threshold(img_gs,127,255,cv2.THRESH_BINARY)
    img_copy=np.ones((thresh1.shape),dtype=thresh1.dtype)
    # ret2,thresh1 = cv2.threshold(img_gs,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plt.subplot(223)
    # plt.imshow(thresh1,'gray')
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(41,41))
    thresh1_closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    thresh1_opening = cv2.morphologyEx(thresh1_closing, cv2.MORPH_OPEN, kernel)
    # thresh1_=cv2.morphologyEx(thresh1_closing, cv2.MORPH_ERODE, kernel)
    # plt.subplot(224)
    # plt.imshow(closing,'gray')
    # plt.show()

    ret,thresh2 = cv2.threshold(img_gs,210,255,cv2.THRESH_BINARY_INV) #205
    thresh2_closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    thresh2_opening = cv2.morphologyEx(thresh2_closing, cv2.MORPH_OPEN, kernel)
    
    aretefact_mask=cv2.bitwise_and(thresh1,thresh2_opening)
    aretefact_mask=cv2.bitwise_not(aretefact_mask)

    # plt.imshow(aretefact_mask,'gray')
    # plt.show()
    contours, hierarchy = cv2.findContours(aretefact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,0),-1)

    # plt.subplot(131)
    # plt.imshow(thresh1,'gray')
    # plt.subplot(132)
    # plt.imshow(thresh2_opening,'gray')
    # plt.subplot(133)
    # plt.imshow(img_copy,'gray')
    # plt.show()

    print(aretefact_mask.shape,img_copy.shape)
    return img_copy

def remove_unkown_class(img):
    img=hist_eq(img)
    img_unknown,mask=hsv.hsv_sep_unkown(img)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))    
    mask_erode=cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(41,41))    
    mask_dilate = cv2.morphologyEx(mask_erode, cv2.MORPH_DILATE, kernel)
    # mask_dilate=mask
    img_copy=np.ones((mask.shape),dtype=mask.dtype)
    contours, hierarchy = cv2.findContours(mask_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,0),-1)
    # plt.subplot(131)
    # plt.imshow(mask,'gray')
    # plt.subplot(132)
    # plt.imshow(mask_dilate,'gray')
    # plt.subplot(133)
    # plt.imshow(img_copy,'gray')
    # plt.show()
    return img_copy


  
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

# img = cv2.imread('assets/var/1112.jpg')
# img=extract_trap(img)
# remove_artefacts(img)

# img = cv2.imread('assets/var/1261.jpg')
# img=extract_trap(img)
# remove_unkown_class(img)