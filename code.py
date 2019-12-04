import cv2
import numpy as np
# Reading in and displaying our image
img = cv2.imread('ri.jpg', cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ', img.shape)

scale_percent = 40  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# cv2.imshow('Original',resized)

#green channel - to improve the contrast and smooth look
green_channel =resized[:,:,1]
cv2.imshow('green channel',green_channel)

#median filtering
kernel = np.ones((3,3),np.float32)/9
smooth = cv2.filter2D(green_channel,-1,kernel)
cv2.imshow('After smooothing',smooth)


# #dilation
# dilation = cv2.dilate(green_channel,kernel,iterations = 1)
# cv2.imshow('After dilation',dilation)


# top hat image -opening
tophat = cv2.morphologyEx(green_channel, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('After tophat',tophat)
# black hat image-closing
blackhat = cv2.morphologyEx(green_channel, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('After blackhat',blackhat)


cbt = (blackhat+tophat)*3

tophat = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)

i = (blackhat+tophat)*10

cv2.imshow("combination of black hat and top hat", cbt)
cv2.imshow("combination of black hat and top hat with smooth", i)
# cv2.imshow("i",i)







retval, threshold = cv2.threshold(green_channel, 60, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold',threshold)



th = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('Adaptive threshold(gaussian) with smooth',th)

th = cv2.adaptiveThreshold(green_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('Adaptive threshold(gaussian) no smooth',th)

th = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('Adaptive threshold(mean) with smooth',th)

th = cv2.adaptiveThreshold(green_channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('Adaptive threshold(mean) no smooth',th)
# gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
# retval2,threshold2 = cv2.threshold(green_channel,220,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('Otsu threshold',threshold2)


cv2.waitKey(0)
cv2.destroyAllWindows()
