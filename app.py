from flask import Flask
from flask import request
from flask import jsonify
import os
import base64
import jsonpickle
import cv2
import json
import time
import numpy as np
from  matplotlib import pyplot as plt
import face_embedding

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png','jpg','bmp'])
@app.route("/")
def home():
    return "Hello Flask 2"

@app.route("/test")
def test():
    return "This is Test"
@app.route("/reco", methods=['POST'])
def get_frame():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)


    #use python program to send http message and get recognization outcome back
    answer = face_embedding.face_reco(img)
    return answer
def allow_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ =="__main__":
    app.run()