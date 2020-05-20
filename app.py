from flask import Flask
from flask import request
from flask import jsonify
import os
import base64
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
    f = request.files.get('file')
    if not allow_file(f.filename):
        return "0000"
    np_img = plt.imread(f)
    answer = face_embedding.face_reco(np_img)
    return answer
def allow_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ =="__main__":
    app.run()