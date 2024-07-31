import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
import keras
from keras import utils as image
from keras import models
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

@app.route('/')
def index():
    return 'Hello, World!!!'

@app.route('/api/upload', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        yolo_model = YOLO('best.pt')

        cls_model = models.load_model('efficientnetv2b3_model.h5', compile=False)
        cls_model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy']
        )

        # Read image file
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Predict with YOLOv8
        results = yolo_model(img)

        # Get bounding boxes
        boxes = results[0].boxes.xyxy  # Get bounding boxes in xyxy format
        predictions = []

        # Loop through each detection and crop
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4].tolist())  # Convert coordinates to integers
            crop = img[y1:y2, x1:x2]

            # Resize and preprocess cropped image
            crop_resized = cv2.resize(crop, (320, 320))
            crop_array = image.img_to_array(crop_resized)
            crop_array = np.expand_dims(crop_array, axis=0)
            crop_array = np.vstack([crop_array])

            # Predict with MobileNetV2
            preds = cls_model.predict(crop_array)
            preds = np.argmax(preds)

            if preds == 0:
                label = 'Abrasions'
            elif preds == 1:
                label = 'Bruises'
            elif preds == 2:
                label = 'Burns'
            elif preds == 3:
                label = 'Cuts'
            elif preds == 4:
                label = 'Laceration'
            else:
                label = 'not classified'

            # Append predictions
            predictions.append({
                "wound": label,
            })

        if predictions:
            return jsonify(predictions[0])
        else:
            return jsonify({"label": "Not detection"})

    return jsonify({"error": "Invalid file format"}), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run()