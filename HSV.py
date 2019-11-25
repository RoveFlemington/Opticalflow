import cv2

image=cv2.imread('1.jpg')
HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(HSV[y,x])
#th2=cv2.adaptiveThreshold(imagegray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

cv2.namedWindow("imageHSV", 0) #调节窗口尺寸

cv2.imshow("imageHSV",HSV)
cv2.namedWindow('image', 0) #调节窗口尺寸
cv2.imshow('image',image)
cv2.setMouseCallback("imageHSV",getpos)
cv2.waitKey(0)
#print (image(10,10,10))

