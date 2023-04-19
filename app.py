from flask import Flask, request
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
df_detector = load_model('df_detector.hdf5')
face_detector = mp.solutions.face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5)
input_shape = (256, 256, 3)

@app.route('/')
def hello():
    return 'hello'

@app.route('/predict', methods=['POST'])
def predict():
    faces = []
    facePred = {}
    try:
      # {frame: [{id, x, y, w, h}, {...}, {...}, ...], ...}
      # {id: 1/0, ...}
      file = request.files['file']
      filename = secure_filename(file.filename)
      file.save(os.path.join('UPLOAD_FOLDER', filename))
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
            faces.append(newFace)
            facePred[id] = pred
            i += 1
    except Exception as e:
       print(e)
       pass
    os.remove(os.path.join('UPLOAD_FOLDER', filename))
    data = {"faces": faces, "facePred": facePred}
    return data

if __name__ == '__main__':
    app.run(host = '192.168.29.75', port = 5000, debug = True)