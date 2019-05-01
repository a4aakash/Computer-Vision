import face_recognition
import cv2
import numpy as np


video_capture = cv2.VideoCapture(0)

# Load sample
image1 = face_recognition.load_image_file("Your Image Here with extension")
encoding1 = face_recognition.face_encodings(image1)[0]

known_face_encodings = [encoding1,
                        #encoding2]
                        ]
known_face_names = ["Put Your Name Which U Wants to See on Ur Image"
                    ,]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)
    rgb_small_frame = frame[:, :, ::-1]# BGR color(OpenCV) to RGB color (face_recognition)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Get all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check if face matches from any saved images
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (a,b,c,d), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #a *= 4
        #b *= 4
        #c *= 4
        #d *= 4

        cv2.rectangle(frame, (d,a), (b, c), (0, 0, 255), 2)
        cv2.rectangle(frame, (d,c), (b,c), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (d,c), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
