import numpy as np
import cv2
img=cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#faceCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
while True:
    ret,frame=img.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces =faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for a,b,c,d in faces:
        cv2.rectangle(frame,(a,b),(a+c,b+d),(0,0,255),2)
    cv2.imshow('video',frame)
    print(frame)
    if cv2.waitKey(1)==ord('q'):
        break
img.release()
cv2.destroyAllWindows()
