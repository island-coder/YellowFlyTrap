import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import utility as util
import preprocessing as pre

def nothing(x):
    pass

# Load image


def hsv_finder(img):      
    # img_macro=cv2.imread('assets/macro_test.jpg')
    # image=img_macro[350:1900,450:1200,:]

    # Create a window
    # img_nesi=cv2.imread('assets/nesi_test.jpg')
    # image=img_nesi[550:2400,2000:2600,:]

    image=img
    h,w=image.shape[0:2]
    h=math.floor(h*0.8)
    w=math.floor(w*0.8)
    print(image.shape[0:2],(h,w))
    image = cv2.resize(image, (h,w))  
    cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

img=cv2.imread('assets/wf_test.jpg')
wf_cropped=img[200:750,2000:2700,:]
# util.viewBGR(img)
# util.viewBGR(wf_cropped)
print(wf_cropped.shape)

img=cv2.imread('assets/macro_test.jpg')
macro_cropped=img[1460:1800,500:1000,:] 
# util.viewBGR(img)
# util.viewBGR(macro_cropped)  
print(macro_cropped.shape)

img=cv2.imread('assets/nesi_test.jpg')
nesi_cropped=img[600:1300,2000:2400,:] 
# util.viewBGR(img)
# util.viewBGR(nesi_cropped)  
print(nesi_cropped.shape)
canvas=np.zeros((700,1600,3),np.uint8)
canvas[0:550,0:700,:]=wf_cropped
canvas[0:340,700:1200,:]=macro_cropped
canvas[0:700,1200:1600,:]=nesi_cropped
plt.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
plt.show()
# canvas=cv2.medianBlur(canvas,7)
canvas=pre.hist_eq(canvas)
plt.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
plt.show()

hsv_finder(canvas)