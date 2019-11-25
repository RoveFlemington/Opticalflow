import numpy as np
import cv2
import math

def rgb2gray(img):
    h=img.shape[0]
    w=img.shape[1]
    img1=np.zeros((h,w),np.uint8)
    for i in range(h):
        for j in range(w):
            img1[i,j]=0.144*img[i,j,0]+0.587*img[i,j,1]+0.299*img[i,j,1]
    return img1

def otsu(img):
    h=img.shape[0]
    w=img.shape[1]
    m=h*w   # 图像像素点总和
    otsuimg=np.zeros((h,w),np.uint8)
    threshold_max=threshold=0   # 定义临时阈值和最终阈值
    histogram=np.zeros(256,np.int32)   # 初始化各灰度级个数统计参数
    probability=np.zeros(256,np.float32)   # 初始化各灰度级占图像中的分布的统计参数
    for i in range (h):
        for j in range (w):
            s=img[i,j]
            histogram[s]+=1   # 统计像素中每个灰度级在整幅图像中的个数
    for k in range (256):
        probability[k]=histogram[k]/m   # 统计每个灰度级个数占图像中的比例
    for i in range (255):
        w0 = w1 = 0   # 定义前景像素点和背景像素点灰度级占图像中的分布
        fgs = bgs = 0   # 定义前景像素点灰度级总和and背景像素点灰度级总和
        for j in range (256):
            if j<=i:   # 当前i为分割阈值
                w0+=probability[j]   # 前景像素点占整幅图像的比例累加
                fgs+=j*probability[j]
            else:
                w1+=probability[j]   # 背景像素点占整幅图像的比例累加
                bgs+=j*probability[j]
        u0=fgs/w0   # 前景像素点的平均灰度
        u1=bgs/w1   # 背景像素点的平均灰度
        g=w0*w1*(u0-u1)**2   # 类间方差
        if g>=threshold_max:
            threshold_max=g
            threshold=i
    print(threshold)
    for i in range (h):
        for j in range (w):
            if img[i,j]>threshold:
                otsuimg[i,j]=255
            else:
                otsuimg[i,j]=0
    return otsuimg

# image = cv.imread("D:/selina.png")
# grayimage = rgb2gray(image)
# otsuimage = otsu(grayimage)
# cv.imshow("image", image)
# cv.imshow("grayimage",grayimage)
# cv.imshow("otsu", otsuimage)
# cv.waitKey(0)
# cv.destroyAllWindows()



# cap = cv2.VideoCapture(0)       # 0为摄像头，中间可以写成视频文件
cap = cv2.VideoCapture("2.mp4")
# 获取第一帧
ret, frame1 = cap.read()

image = cv2.imread(frame1)
grayimage = rgb2gray(image)
otsuimage = otsu(grayimage)
# cv2.imshow("image", image)
# cv2.imshow("grayimage",grayimage)
# cv2.imshow("otsu", otsuimage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)    #实现由RGB向HSV通道转变
hsv = np.zeros_like(frame1)


# mask = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
# lower = np.array([10, 46, 40])
# upper = np.array([20, 160, 90])
# prvs = cv2.inRange(mask, lower, upper)
# hsv = np.zeros_like(frame1)

#遍历每一行的第1列
# hsv[...,1] = 255
s=0

while(1):
    ret, frame2 = cap.read()
    #
    # nextmask = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    # next = cv2.inRange(nextmask, lower, upper)

    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    #返回一个两通道的光流向量，实际上是每个点的像素位移值
    flow = cv2.calcOpticalFlowFarneback(otsuimage,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # // _prev0：输入前一帧图像
    # // _next0：输入后一帧图像
    # // _flow0：输出的光流
    # // pyr_scale：金字塔上下两层之间的尺度关系
    # // levels：金字塔层数
    # // winsize：均值窗口大小，越大越能denoise并且能够检测快速移动目标，但会引起模糊运动区域
    # // iterations：迭代次数
    # // poly_n：像素领域大小，一般为5，7
    # // poly_sigma：高斯标注差，一般为1 - 1.5
    # // flags：计算方法。主要包括OPTFLOW_USE_INITIAL_FLOW和OPTFLOW_FARNEBACK_GAUSSIAN

    #print(flow.shape)
    x = abs(flow.sum(axis=1))
    zx = abs(x.sum())
    y = abs(flow.sum(axis=0))
    zy = abs(y.sum())

    zz=(abs(math.sqrt(zx*zx+zy*zy))/10000)
    # print(flow)
    s=int(s+zz)
    print(s)

# 输出的光流矩阵。矩阵大小同输入的图像一样大1920*1080，但是矩阵中的每一个元素可不是一个值，而是两个值，分别表示这个点在x方向与y方向的运动量（偏移量）

    #笛卡尔坐标转换为极坐标，获得极轴和极角
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(100) & 0xff                     # waitkey里面的数值就是读取视频时的帧速度，单位为毫秒
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    # s=(s+zz)/10000
    # print(s)
    # prvs = next
    otsuimage = next

cap.release()
cv2.destroyAllWindows()

