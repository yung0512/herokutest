import cv2
import numpy as np
import dlib
import csv
from skimage import io
import pandas as pd
import os
import base64


#the detector of the human frontal face
detector = dlib.get_frontal_face_detector()
#dlib face predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#dlib face recogniation model
face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

"""
def function image_reco(image)
1.將已經確認的人臉放入資料庫
2.讀入從htttp傳來的照片
3.傳回辨識結果
"""

#return the face 128D_feature
def coculate_128D_features():
    people = os.listdir("database/")
    people.sort()
    name_list = []
    person_num = len(people)
    with open("features_of_database.csv","w",newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in people:
            face_descriptor = 0
            img_rd = io.imread("database/"+person)
            faces = detector(img_rd)
            print("image with faces detected")

            #to make sure that at least one face in the image
            if len(faces) != 0 :
                shape = predictor(img_rd,faces[0])
                face_descriptor = face_rec.compute_face_descriptor(img_rd,shape)
                writer.writerow(face_descriptor)
                name_list.append(person[0:len(person)-4])
            else:
                face_descriptor = 0
                print("no face")

    return name_list #return the name list in the database
def euclidean_distance(feature_1,feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1-feature_2)))
    return dist
def get_namelist():
    people = os.listdir("database/")
    people.sort()
    name_list = []
    person_num = len(people)
    for person in people:
        name_list.append(person[0:len(person)-4])
    return name_list

def face_reco(http_image):#接收app.py已經轉好的RGB numpy array
    #將database的照片label並放入資料庫裡面
    """
    known_face_encodings = []
    known_face_names = []
    people = os.listdir("database/")
    people.sort()
    for person in people:
        face_image = face_recognition.load_image_file("database/"+person)
        face_image_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_image_encoding)
        known_face_names.append(person[0:len(person)-4])
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(http_image,cv2.COLOR_BGR2RGB)

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
    """
   
    if os.path.exists("features_of_database.csv"):
        #read csv file
        path_features_known_csv = "features_of_database.csv"
        csv_rd  = pd.read_csv(path_features_known_csv,header = None)
        faces = detector(http_image,0)
        #to store the known faces 128d_fetures from csv to array
        features_known_arr = []
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            for j in range(0,len(csv_rd.iloc[i])):
                features_someone_arr.append(csv_rd.iloc[i][j])
            features_known_arr.append(features_someone_arr)                    
        
        
        if len(faces)!=0:
            shape = predictor(http_image,faces[0])
            features_image = face_rec.compute_face_descriptor(http_image,shape)
            distance_list = []
            for i in range(len(features_known_arr)):
                distance_tmp = euclidean_distance(features_image,features_known_arr[i])
                #coculate the image euclidean distance with images in database
                distance_list.append(distance_tmp)
            ans_index = distance_list.index(min(distance_list))
            namelist = get_namelist()
            if min(distance_list) < 0.4:
                print("may be "+ namelist[ans_index])
                return namelist[ans_index]
            else:
                print("unknown person")
                return "unkown person"
        else:
            return "no face in image!"        
    else:
        return "file not exist!"       
    
if __name__ == "__main__":
    test = cv2.imread('../FISHEYE/FisheyeCalibration_Src/FisheyeCorrection2/face_reco.jpg')
    #coculate_128D_features()
    answer = face_reco(test)
    print(answer)
