from __future__ import print_function
import requests
import json
import cv2

addr = 'http://127.0.0.1:5000/reco'
test_url = addr 

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('f4006s.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print(response.text)

# expected output: {u'message': u'image received. size=124x124'}