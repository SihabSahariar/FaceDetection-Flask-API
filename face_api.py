from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
app = Flask(__name__)

@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img_bytes = file.read()
    img_bytes1 = base64.b64decode(img_bytes)
    nparr = np.frombuffer(img_bytes1, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform face detection using a face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # Prepare the list of detected faces
    detected_faces = []
    for (x, y, w, h) in faces:
        detected_faces.append({
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        })

    return jsonify({'faces': detected_faces})

if __name__ == '__main__':
    app.run()
