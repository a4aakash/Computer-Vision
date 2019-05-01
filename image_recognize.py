import face_recognition as fr
import cv2

img= fr.load_image_file("pp1.jpg")
img_encode = fr.face_encodings(img)[0]
real = fr.load_image_file("pp1.jpg")
real_encode = fr.face_encodings(real)[0]
result = fr.compare_faces([img_encode], real_encode)
print(result)
