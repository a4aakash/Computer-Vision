import numpy as np
import cv2
img=cv2.VideoCapture(0)
while True:
    ret,frame=img.read()
    cv2.imshow('video',frame)
    ret,frame1=img.read()
    img1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    a=cv2.absdiff(img1,img2)
    ret,frame2=img.read()
    img3=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    b=cv2.absdiff(img2,img3)
    d=cv2.bitwise_and(a,b)
    cv2.imshow('v1',d)
    if cv2.waitKey(1)== ord('q'):
        break
img.release()
cv2.destroyAllWindows()
