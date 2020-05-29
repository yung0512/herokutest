#!/usr/bin/python
#coding=UTF-8
 
"""
TCP/IP Server sample
"""
"""
1.要把server 建在GUI上面
2.GUI窗口要有甚麼
    1.socket 接收的log
    2.人臉辨識前後的圖

"""
import socket
import cv2
import threading
import face_embedding
import time
 
bind_ip = "192.168.31.45"
bind_port = 9999
 
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 
server.bind((bind_ip, bind_port))
global flag 
flag =False       
server.listen(5)
 
print("keep listening")
 
def handle_client(client_socket):
   # img = cv2.imread("../FISHEYE/FisheyeCalibration_Src/FisheyeCorrection2/face_recognize.jpg")
    global flag
    request = client_socket.recv(1024)
    print("recieve from"+str(request.decode()))
    message = str(request.decode())
    #answer = face_embedding.face_reco(img)
    if message=="arm":
        while True:
            if flag:
                answer = "drop"
                client_socket.send(answer.encode())
                flag = False
            time.sleep(1)
    if message=="AGV":
        flag = True
        answer = "ok"
        client_socket.send(answer.encode())
    if message=="speaker":
        if flag:
            answer = "ok"
            client_socket.send(answer.encode())  
            flag = False
    print("send done")
    client_socket.close()
    img = cv2.imread("../FISHEYE/FisheyeCalibration_Src/FisheyeCorrection2/face_recognize.jpg")
    answer = face_embedding.face_reco(img)
    print("the answer is :"+answer)
    
    
while True:
    client, addr = server.accept()
    print("accept connection")
    client_handler = threading.Thread(target=handle_client, args=(client,))
    client_handler.start()
    img = cv2.imread("../FISHEYE/FisheyeCalibration_Src/FisheyeCorrection2/face_reco.jpg")
    cv2.imshow('test',img)

    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break
    