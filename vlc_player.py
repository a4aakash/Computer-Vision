import numpy as np
import cv2
import vlc
ob = vlc.Instance()
player = ob.media_player_new()
Media = ob.media_new('your video file with extension')
player.set_media(Media)
k,flag=0,0
img=cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
     
  #  player.play()
    ret,frame=img.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces =faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for a,b,c,d in faces:
        cv2.rectangle(frame,(a,b),(a+c,b+d),(0,0,255),2)
    #cv2.imshow('video',frame)
    
    if len(faces)!=0:
        flag=1
    else:
        flag=0
    if flag==1:
        k=1
        player.play()
    if flag==0 and k==1:
        k=0
        player.pause()

    print(faces)
   # if cv2.waitKey(1)==ord('q'):
    #    break
img.release()
cv2.destroyAllWindows()

#-----------------------------------------------------------------

# Playing a video from vlc-player

'''ob = vlc.Instance()
player = ob.media_player_new()
Media = ob.media_new('one.mp4')
player.set_media(Media)
player.play()'''


#-----------------------------------------------------------------
