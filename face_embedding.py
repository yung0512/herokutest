import face_recognition
import cv2
import numpy as np
import os
import base64
# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
"""
Amy_image = face_recognition.load_image_file("database/Amy.jpg")
Amy_face_encoding = face_recognition.face_encodings(Amy_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]
"""
"""
def function image_reco(image)
1.將已經確認的人臉放入資料庫
2.讀入從htttp傳來的照片
3.傳回辨識結果
"""
known_face_encodings = []
known_face_names = []
people = os.listdir("database/")
people.sort()
for person in people:
    face_image = face_recognition.load_image_file(person)
    face_image_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_image)
    known_face_names.append(face_image_encoding)


def face_reco(http_image):#接收app.py已經轉好的RGB numpy array
    #將database的照片label並放入資料庫裡面
    known_face_encodings = []
    known_face_names = []
    people = os.listdir("database/")
    people.sort()
    for person in people:
        face_image = face_recognition.load_image_file(person)
        face_image_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_image)
        known_face_names.append(face_image_encoding)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = http_image[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(http_image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(http_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(http_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    return name