import os
filepath = "../FISHEYE/FisheyeCalibration_Src/FisheyeCorrection2/face_reco.jpg"

# 檢查檔案是否存在
if os.path.isfile(filepath):
  print("檔案存在。")
else:
  print("檔案不存在。")