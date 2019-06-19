import cv2 as cv
import numpy as np
cap = cv.VideoCapture("./a041-0536C.mp4")
# cap = cv.VideoCapture("./a001-0487C.mp4")
# cap = cv.VideoCapture("./a001-0488C.mp4")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255                                                    # saturation purity how much white
while(1):
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    prvs = cv.medianBlur(prvs,5)
    next = cv.medianBlur(next,5)
    # print(prvs.shape,next.shape)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5,3,7,4,7,5, 0)
    # print(flow.shape)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    
    # for i in range(224):
    #     for j in range(224):
    #         if mag[i][j] < 2:
    #             mag[i][j] = 0
    mag = (mag>1) * mag
    # ang = np.logical_or(np.logical_and((ang>1),(ang<2)),np.logical_and((ang>4),(ang<5))) * ang
    hsv[...,0] = ang*180/np.pi/2                                     #hue   which colour  draw for purple and yellow
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)                # intensity brightness draw for extra bright
    
    # for i in range(224):
    #     for j in range(224):
    #         #if hsv[i][j][0] > 60 and hsv[i][j][0]<120):
    #         if hsv[i][j][0] not in range(60,120) or hsv[i][j][0] not in range(240,300):
    #             hsv[i][j] = 0
    #         # else:
    #         #     hsv[i][j][0] = 0

    # for i in range(224):
    #     for j in range(224):
    #         if hsv[i][j][2] < 100:
    #             hsv[i][j][2] = 0

    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    bgr = cv.medianBlur(bgr,5)
    bgr = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
    print(bgr.shape)
    # ret,bgr = cv.threshold(bgr,50,255,cv.THRESH_BINARY)
    #print(cap.get(3),cap.get(4))
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    prvs = next
cap.release()
cv.destroyAllWindows()
# print(mag)
# print('MAG',mag,mag.shape)
# print('ANG',ang)
#print('HSV',hsv,sum(hsv[...,2]))
# for i in range(224):
#     for j in range(224):
#         print(hsv[i][j][2],end=' ')
#     print()
mag = np.reshape(mag,(224,224,1))
# print(mag)
ang = np.reshape(ang,(224,224,1))
mod_flow = np.concatenate((mag,ang),axis=-1)