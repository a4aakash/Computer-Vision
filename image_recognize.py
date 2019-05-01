import face_recognition as fr
import cv2

img= fr.load_image_file("Put Your image with extension")
img_encode = fr.face_encodings(img)[0]
real = fr.load_image_file("Put Image For check with extension")
real_encode = fr.face_encodings(real)[0]
result = fr.compare_faces([img_encode], real_encode)
print(result)
