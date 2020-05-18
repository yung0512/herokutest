from flask import Flask
from flask import request
import os
import base64
import cv2
import numpy as np
import face_embedding

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello Flask 2"

@app.route("/test")
def test():
    return "This is Test"
@app.route("/reco", methods=['POST'])
def get_frame():
    upload_file = request.files['file']
    img = upload_file.read()
    image = cv2.imdecode(np.frombuffer(img,np.uint8),cv2.IMREAD_COLOR)
    reco_outcome = face_embedding.face_reco(image)
    return reco_outcome
if __name__ =="__main__":
    app.run()