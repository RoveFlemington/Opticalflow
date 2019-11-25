import numpy as np
import cv2
import math

# cap = cv2.VideoCapture(0)       # 0为摄像头，中间可以写成视频文件
cap = cv2.VideoCapture("1.mp4")
# 获取第一帧
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)    # 实现由RGB向HSV通道转变
hsv = np.zeros_like(frame1)

# 遍历每一行的第1列
hsv[...,1] = 255
s=0

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # 返回一个两通道的光流向量，实际上是每个点的像素位移值
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

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

    # print(flow.shape)
    x = abs(flow.sum(axis=1))
    zx = abs(x.sum())
    y = abs(flow.sum(axis=0))
    zy = abs(y.sum())

    zz = (abs(math.sqrt(zx*zx+zy*zy))/10000)
    # print(flow)
    s = int(s+zz)
    print(s)

# 输出的光流矩阵。矩阵大小同输入的图像一样大1920*1080，但是矩阵中的每一个元素可不是一个值，而是两个值，分别表示这个点在x方向与y方向的运动量（偏移量）

    # 笛卡尔坐标转换为极坐标，获得极轴和极角
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
    prvs = next


cap.release()
cv2.destroyAllWindows()
