import numpy as np
import cv2
import matplotlib.pyplot  as plt

def otsuCompute(grayImg):
    # 类间方差
    g = 0
    # 遍历矩阵
    for i in range(0,256):
        # 背景像素点占整幅图像的比例

        W0 = 0.0
        w0 = 0
        # 背景图像平均灰度
        U0 = 0
        u0 = 0.0
        # 前景图像占整幅图像的比例
        W1 = 0.0
        w1 = 0
        # 前景图像的平均灰度
        U1 = 0
        u1 = 0.0


        for element in grayImg.flat:

            # 大于i为前景
            if element>i:
                w1+=1
                u1+=element
            else:
                w0+=1
                u0+=element
        try:
            W0 = w0/(w0+w1)
            W1 = 1 - W0
            U0 = u0/w0
            U1 = u1/w1

            if W1*W0*((U0 - U1)**2) > g:
                g=W1*W0*((U0 - U1)**2)
                thres=i
        except:
            pass
    return  thres

img=cv2.imread('1.jpg',cv2.IMREAD_UNCHANGED)
grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.medianBlur(grayImg,5,grayImg)
# 大津算法二值化分割前景和背景
ret,thresOtsu=cv2.threshold(grayImg,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
ret1,thresMyOtsu=cv2.threshold(grayImg,otsuCompute(grayImg),255,cv2.THRESH_BINARY_INV)



print('Otsu threshold is {0}'.format(ret))
print('my Otsu threshold is {0}'.format(ret1))
cv2.imshow('1',img)
cv2.imshow('Otsu',thresOtsu)
cv2.imshow('my Otsu',thresMyOtsu)
cv2.waitKey()
cv2.destroyAllWindows()

