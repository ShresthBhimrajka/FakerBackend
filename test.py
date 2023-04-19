import cv2
from tensorflow.keras.models import load_model
import os
import time
import mediapipe as mp
# import face_detection

# modelFile = "save_dir_1\ee5e34e4fcf86a7c5174c32dd67569dd3e4dfe9f\MobileNetSSD_deploy.caffemodel"
# configFile = "deploy.prototxt.txt"
# net = cv2.dnn.readNetFromCaffe(prototxt="deploy.prototxt", caffeModel="res10_300x300_ssd_iter_140000_fp16.caffemodel")
df_detector = load_model('df_detector.hdf5')
face_detector = mp.solutions.face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5)
# face_detector = cv2.CascadeClassifier('face_detector.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# face_detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

# def predict(): # face_detection
#     start = time.time()
#     faces = []
#     facePred = {}
#     try:
#       # {frame: [{id, x, y, w, h}, {...}, {...}, ...], ...}
#       # {id: 1/0, ...}
#       filename = 'bb5e30ee-3bb6-4c55-a8f3-c8233fc9bf71.jpeg'
#       img = cv2.imread(os.path.join('UPLOAD_FOLDER', filename))[:, :, ::-1]
#       facesDet = face_detector.detect(img)
#       i = 0
#       for (xmin, ymin, xmax, ymax, detection_confidence) in facesDet:
#         if(detection_confidence > 0.7):
#             x1 = int(xmin)
#             y1 = int(ymin)
#             x2 = int(xmax)
#             y2 = int(ymax)
#             crop_img = img[y1:y2, x1:x2]
#             resized_img = cv2.resize(crop_img, (256, 256)).reshape((1, 256, 256, 3))
#             pred = int(df_detector.predict(resized_img)[0][0])
#             id = filename + str(i)
#             newFace = {"id": id, "x": int(x1), "y": int(y1), "width": int(xmax - xmin), "height": int(ymax - ymin)}
#             print(newFace)
#             faces.append(newFace)
#             facePred[id] = pred
#             i += 1
#     except Exception as e:
    #    print(e)
    #    pass
#     print("faces", faces, "\nfacePred",facePred)
#     end = time.time()
#     print((end - start) * 1000, "ms")

# def predict(): # cv2_dnn
#     faces = []
#     facePred = {}
#     try:
#       # {frame: [{id, x, y, w, h}, {...}, {...}, ...], ...}
#       # {id: 1/0, ...}
#         filename = 'bb5e30ee-3bb6-4c55-a8f3-c8233fc9bf71.jpeg'
#         image = cv2.imread(os.path.join('UPLOAD_FOLDER', filename))
#         image_height, image_width, _ = image.shape
#         preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 117.0, 123.0))
#         net.setInput(preprocessed_image)
#         facesDet = net.forward()
#         i = 0
#         for face in facesDet[0][0]:
#             confidence = face[2]
#             if(confidence > 0.5):
#                 bbox = face[3:]
#                 x1 = int(bbox[0] * image_width)
#                 y1 = int(bbox[1] * image_height)
#                 x2 = int(bbox[2] * image_width)
#                 y2 = int(bbox[3] * image_height)
#                 crop_img = image[y1:y2, x1:x2]
#                 cv2.imshow("image", crop_img)
#                 cv2.waitKey(0)
#                 resized_img = cv2.resize(crop_img, (256, 256)).reshape((1, 256, 256, 3))
#                 pred = int(df_detector.predict(resized_img)[0][0])
#                 id = filename + str(i)
#                 newFace = {"id": id, "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
#                 print(newFace)
#                 faces.append(newFace)
#                 facePred[id] = pred
#                 i += 1
#     except Exception as e:
    #    print(e)
    #    pass
#     print("faces", faces, "\nfacePred",facePred)

# def predict(): # cv2_haarcascade
#     faces = []
#     facePred = {}
#     try:
#       # {frame: [{id, x, y, w, h}, {...}, {...}, ...], ...}
#       # {id: 1/0, ...}
#         filename = 'bb5e30ee-3bb6-4c55-a8f3-c8233fc9bf71.jpeg'
#         img = cv2.imread(os.path.join('UPLOAD_FOLDER', filename))
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         facesDet = face_detector.detectMultiScale(gray, 1.5, 8)
#         i = 0
#         for (x, y, w, h) in facesDet:
#                 x1 = int(x)
#                 y1 = int(y)
#                 x2 = int(x + w)
#                 y2 = int(y + h)
#                 crop_img = img[y1:y2, x1:x2]
#                 eyes = eye_cascade.detectMultiScale(crop_img, 1.1, 3)
#                 # if len(eyes) == 0:
#                 #      continue
#                 cv2.imshow("image", crop_img)
#                 cv2.waitKey(0)
#                 resized_img = cv2.resize(crop_img, (256, 256)).reshape((1, 256, 256, 3))
#                 pred = int(df_detector.predict(resized_img)[0][0])
#                 id = filename + str(i)
#                 newFace = {"id": id, "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
#                 print(newFace)
#                 faces.append(newFace)
#                 facePred[id] = pred
#                 i += 1
#     except Exception as e:
    #    print(e)
    #    pass
#     print("faces", faces, "\nfacePred",facePred)

def predict(): # mediapipe
    start = time.time()
    faces = []
    facePred = {}
    try:
      # {frame: [{id, x, y, w, h}, {...}, {...}, ...], ...}
      # {id: 1/0, ...}
      filename = 'bb5e30ee-3bb6-4c55-a8f3-c8233fc9bf71.jpeg'
      img = cv2.imread(os.path.join('UPLOAD_FOLDER', filename))[:,:,::-1]
      img_height, img_width, _ = img.shape
      facesDet = face_detector.process(img)
      i = 0
      for face in facesDet.detections:
            face_data = face.location_data.relative_bounding_box
            x1 = int(face_data.xmin * img_width)
            y1 = int(face_data.ymin * img_height)
            width = int(face_data.width * img_width)
            height = int(face_data.height * img_height)
            x2 = int(x1 + width)
            y2 = int(y1 + height)
            crop_img = img[y1:y2, x1:x2]
            resized_img = cv2.resize(crop_img, (256, 256)).reshape((1, 256, 256, 3))
            pred = int(df_detector.predict(resized_img)[0][0])
            id = filename + str(i)
            newFace = {"id": id, "x": x1, "y": y1, "width": width, "height": height}
            print(newFace)
            faces.append(newFace)
            facePred[id] = pred
            i += 1
    except Exception as e:
       print(e)
       pass
    print("faces", faces, "\nfacePred",facePred)
    end = time.time()
    print((end - start) * 1000, "ms")

predict()